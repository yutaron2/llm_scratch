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
from tokenizer.bpe import BPETokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_text(path: str) -> str:
    return (ROOT_DIR / path).read_text(encoding="utf-8")


def resolve_artifact_path(directory: str, filename: str) -> Path:
    return ROOT_DIR / directory / filename


def load_tokenizer(directory: str, filename: str) -> BPETokenizer:
    tokenizer_path = resolve_artifact_path(directory, filename)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            "Tokenizer artifact was not found. "
            f"Expected: {tokenizer_path}. "
            "Run src/train_tokenizer.py first."
        )
    return BPETokenizer.load(str(tokenizer_path))


def save_checkpoint(model, tokenizer, checkpoint_dir: Path, epoch: int) -> None:
    version_dir = checkpoint_dir / f"model_{epoch}"
    version_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), version_dir / "model.pth")
    tokenizer.save(str(version_dir / "tokenizer.json"))


def test_prediction(model, tokenizer, input_text, seq_len, temperature=0.0):
    input_ids = tokenizer.encode(input_text)
    if len(input_ids) < 1:
        raise ValueError("At least 1 token is required for inference with the current model.")

    if len(input_ids) > seq_len:
        input_ids = input_ids[-seq_len:]

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model(input_tensor)
        last_token_probs = output[0, -1, :]

        if temperature <= 0:
            predicted_token_id = int(torch.argmax(last_token_probs).item())
        else:
            probs = torch.softmax(last_token_probs / temperature, dim=-1)
            predicted_token_id = int(torch.multinomial(probs, num_samples=1).item())

        return tokenizer.decode([predicted_token_id])


def generate_seq(model, tokenizer, text, seq_len, count=0, temperature=0.0):
    next_token = test_prediction(model, tokenizer, text, seq_len, temperature=temperature)
    if count < 20:
        return generate_seq(model, tokenizer, text + next_token, seq_len, count + 1, temperature)
    return text + next_token


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    text = load_text(cfg.data.input_path)
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )

    print(
        "Loaded tokenizer from: "
        f"{resolve_artifact_path(cfg.artifacts.tokenizers_dir, cfg.artifacts.tokenizer_filename)}"
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    if tokenizer.merges:
        print(f"最初のマージ: {tokenizer.describe_merge(0)}")

    token_ids = tokenizer.encode(text)
    train_inputs, train_targets = create_training_data(
        token_ids=token_ids,
        seq_len=cfg.training.sequence_length,
        device=device,
    )
    sample_index = min(20, len(train_inputs) - 1)

    print(f"学習データ数: {len(train_inputs)}")
    print(f"例 - 入力: '{tokenizer.decode(train_inputs[sample_index].tolist())}'")
    print(f"例 - 正解: '{tokenizer.decode(train_targets[sample_index].tolist())}'")

    model = SimpleGPTPredictor(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length,
        num_layers=cfg.model.num_layers,
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
    criterion = nn.CrossEntropyLoss()

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
            target_batch = train_targets[index : index + batch_size]

            output = model(input_batch)
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), target_batch.reshape(-1))
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
