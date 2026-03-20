import math
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from datasets.text_dataset import create_training_data
from models.simple_transformer import SimpleGPTPredictor, device
from tokenizer.artifacts import (
    load_text,
    load_tokenizer,
    resolve_tokenizer_artifact_path,
    validate_loaded_tokenizer,
)


ROOT_DIR = Path(__file__).resolve().parent.parent


def save_checkpoint(model, tokenizer, checkpoint_dir: Path, epoch: int) -> None:
    version_dir = checkpoint_dir / f"model_{epoch}"
    version_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), version_dir / "model.pth")
    tokenizer.save(str(version_dir / "tokenizer.json"))


def build_decoder_inputs(next_token_targets: torch.Tensor, bos_token_id: int) -> torch.Tensor:
    bos_column = torch.full(
        (next_token_targets.size(0), 1),
        bos_token_id,
        device=next_token_targets.device,
        dtype=next_token_targets.dtype,
    )
    return torch.cat((bos_column, next_token_targets[:, :-1]), dim=1)


def build_inference_decoder_input(source_ids: list[int], bos_token_id: int) -> list[int]:
    if not source_ids:
        raise ValueError("source_ids must contain at least one token")
    return [bos_token_id, *source_ids[1:]]


def predict_next_token(model, tokenizer, input_text, seq_len, temperature=0.0):
    source_ids = tokenizer.encode(input_text)
    if len(source_ids) < 1:
        raise ValueError("At least 1 token is required for inference with the current model.")

    if len(source_ids) > seq_len:
        source_ids = source_ids[-seq_len:]

    source_tensor = torch.tensor([source_ids], device=device)
    source_padding_mask = model.make_padding_mask(source_tensor)
    decoder_input_ids = build_inference_decoder_input(source_ids, tokenizer.bos_token_id)
    decoder_input = torch.tensor([decoder_input_ids], device=device)
    decoder_padding_mask = model.make_padding_mask(decoder_input)

    with torch.no_grad():
        memory = model.encode(source_tensor, src_key_padding_mask=source_padding_mask)
        output = model.decode(
            memory,
            decoder_input,
            memory_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=decoder_padding_mask,
        )
        next_token_logits = output[0, -1, :]

        if temperature <= 0:
            predicted_token_id = int(torch.argmax(next_token_logits).item())
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            predicted_token_id = int(torch.multinomial(probs, num_samples=1).item())

        if predicted_token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
            return ""
        return tokenizer.decode([predicted_token_id])


def generate_seq(model, tokenizer, text, seq_len, count=0, temperature=0.0):
    next_token = predict_next_token(model, tokenizer, text, seq_len, temperature=temperature)
    if not next_token or count >= 20:
        return text + next_token
    return generate_seq(model, tokenizer, text + next_token, seq_len, count + 1, temperature)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    text = load_text(cfg.data.input_path)
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )
    validate_loaded_tokenizer(tokenizer, text, cfg.tokenizer.vocab_size)

    print(
        "Loaded tokenizer from: "
        f"{resolve_tokenizer_artifact_path(cfg.artifacts.tokenizers_dir, cfg.artifacts.tokenizer_filename)}"
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    if tokenizer.merges:
        print(f"最初のマージ: {tokenizer.describe_merge(0)}")

    token_ids = tokenizer.encode(text)
    train_inputs, next_token_targets = create_training_data(
        token_ids=token_ids,
        seq_len=cfg.training.sequence_length,
        device=device,
    )
    decoder_inputs = build_decoder_inputs(
        next_token_targets=next_token_targets,
        bos_token_id=tokenizer.bos_token_id,
    )
    labels = next_token_targets
    sample_index = min(20, len(train_inputs) - 1)

    print(f"学習データ数: {len(train_inputs)}")
    print(f"例 - エンコーダ入力: '{tokenizer.decode(train_inputs[sample_index].tolist())}'")
    print(f"例 - デコーダ入力: '{tokenizer.decode(decoder_inputs[sample_index].tolist())}'")
    print(f"例 - ラベル: '{tokenizer.decode(labels[sample_index].tolist())}'")

    model = SimpleEncoderDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length,
        num_layers=cfg.model.num_layers,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.scheduler.t_max,
        eta_min=cfg.scheduler.eta_min,
    )

    checkpoint_dir = ROOT_DIR / cfg.artifacts.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n学習開始...")

    for epoch in range(cfg.training.epochs):
        total_loss = 0.0
        batch_size = cfg.training.batch_size
        num_batches = math.ceil(len(train_inputs) / batch_size)

        for index in tqdm(range(0, len(train_inputs), batch_size)):
            optimizer.zero_grad()

            input_batch = train_inputs[index : index + batch_size]
            decoder_input_batch = decoder_inputs[index : index + batch_size]
            label_batch = labels[index : index + batch_size]

            output = model(input_batch, decoder_input_batch)
            loss = nn.functional.cross_entropy(
                output.reshape(-1, output.size(-1)),
                label_batch.reshape(-1),
                ignore_index=model.pad_token_id,
            )
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {current_lr:.6e}")
        scheduler.step()

        save_checkpoint(model, tokenizer, checkpoint_dir, epoch)


if __name__ == "__main__":
    main()
