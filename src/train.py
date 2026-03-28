from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig

from datasets.text_dataset import create_autoregressive_dataloader
from models.simple_decoder_transformer import SimpleDecoderTransformer
from training.optimization import build_optimizer, build_scheduler
from training.trainer import Trainer
from tokenizer.artifacts import load_text, load_tokenizer
from utils.model import get_parameter_counts


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_sample_batch(tokenizer, batch) -> None:
    inputs = batch["inputs"]
    labels = batch["labels"]
    sample_index = min(20, len(inputs) - 1)
    logger.info("Training samples: {} in preview batch", len(inputs))
    logger.info(
        "Example decoder input: {!r}",
        tokenizer.decode(inputs[sample_index].tolist(), skip_special_tokens=False),
    )
    logger.info(
        "Example label: {!r}",
        tokenizer.decode(labels[sample_index].tolist(), skip_special_tokens=False),
    )


def load_token_ids(input_path: str, tokenizer, split_name: str):
    split_label = split_name.capitalize()
    logger.info("Loading {} corpus from: {}", split_name, input_path)
    text = load_text(input_path)
    logger.info("Tokenizing {} corpus...", split_name)
    token_ids = tokenizer.encode(text)
    logger.info("{} corpus length: {} tokens", split_label, len(token_ids))
    return token_ids


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    logger.info("Using device: {}", DEVICE)

    logger.info("Loading tokenizer artifact...")
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )
    logger.info("Tokenizer vocab size: {}", tokenizer.vocab_size)

    train_token_ids = load_token_ids(
        cfg.data.train,
        tokenizer,
        "training",
    )
    validation_token_ids = load_token_ids(
        cfg.data.val,
        tokenizer,
        "validation",
    )

    if cfg.data.val == cfg.data.train:
        logger.info(
            "Validation uses the same corpus as training. "
            "This run is measuring memorization/overfitting.",
        )

    logger.info("Building decoder-only autoregressive dataloader...")
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
    logger.info("Training windows: {}", len(train_loader.dataset))
    logger.info("Validation windows: {}", len(validation_loader.dataset))

    logger.info("Building model...")
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
    logger.info(
        "Model parameters: "
        "total={:,} "
        "trainable={:,} "
        "frozen={:,}",
        parameter_counts.total,
        parameter_counts.trainable,
        parameter_counts.non_trainable,
    )

    optimizer = build_optimizer(model, cfg.training.optimizer)
    logger.info("Using optimizer: {}", optimizer.__class__.__name__)

    scheduler = build_scheduler(optimizer, cfg.training.get("scheduler"))
    if scheduler is None:
        logger.info("Learning rate scheduler: disabled")
    else:
        scheduler_interval = cfg.training.scheduler.get("interval", "epoch")
        logger.info(
            "Learning rate scheduler: {} ({})",
            scheduler.__class__.__name__,
            scheduler_interval,
        )

    checkpoint_dir = ROOT_DIR / cfg.artifacts.checkpoints_dir
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        device=DEVICE,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
