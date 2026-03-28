from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig

from datasets.text_dataset import create_autoregressive_dataloader
from models.simple_decoder_transformer import SimpleDecoderTransformer
from training.trainer import Trainer
from tokenizer.artifacts import load_text, load_tokenizer
from utils.model import get_parameter_counts


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_sample_batch(tokenizer, batch) -> None:
    inputs = batch["inputs"]
    labels = batch["labels"]
    sample_index = min(20, len(inputs) - 1)
    print(f"Training samples: {len(inputs)} in preview batch", flush=True)
    print(
        "Example decoder input: "
        f"{tokenizer.decode(inputs[sample_index].tolist(), skip_special_tokens=False)!r}",
        flush=True,
    )
    print(
        "Example label: "
        f"{tokenizer.decode(labels[sample_index].tolist(), skip_special_tokens=False)!r}",
        flush=True,
    )


def load_token_ids(input_path: str, tokenizer, split_name: str):
    split_label = split_name.capitalize()
    print(f"Loading {split_name} corpus from: {input_path}", flush=True)
    text = load_text(input_path)
    print(f"Tokenizing {split_name} corpus...", flush=True)
    token_ids = tokenizer.encode(text)
    print(f"{split_label} corpus length: {len(token_ids)} tokens", flush=True)
    return token_ids


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    print(f"Using device: {DEVICE}", flush=True)

    print("Loading tokenizer artifact...", flush=True)
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}", flush=True)

    train_token_ids = load_token_ids(
        cfg.data.input_path,
        tokenizer,
        "training",
    )
    validation_token_ids = load_token_ids(
        cfg.validation.input_path,
        tokenizer,
        "validation",
    )

    if cfg.validation.input_path == cfg.data.input_path:
        print(
            "Validation uses the same corpus as training. "
            "This run is measuring memorization/overfitting.",
            flush=True,
        )

    print("Building decoder-only autoregressive dataloader...", flush=True)
    train_loader = create_autoregressive_dataloader(
        token_ids=train_token_ids,
        seq_len=cfg.training.sequence_length,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )
    validation_loader = create_autoregressive_dataloader(
        token_ids=validation_token_ids,
        seq_len=cfg.training.sequence_length,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )
    preview_batch = next(iter(train_loader))
    log_sample_batch(tokenizer, preview_batch)
    print(f"Training windows: {len(train_loader.dataset)}", flush=True)
    print(f"Validation windows: {len(validation_loader.dataset)}", flush=True)

    print("Building model...", flush=True)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.to(DEVICE)
    model.train()
    parameter_counts = get_parameter_counts(model)
    print(
        "Model parameters: "
        f"total={parameter_counts.total:,} "
        f"trainable={parameter_counts.trainable:,} "
        f"frozen={parameter_counts.non_trainable:,}",
        flush=True,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    checkpoint_dir = ROOT_DIR / cfg.artifacts.checkpoints_dir
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        device=DEVICE,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
