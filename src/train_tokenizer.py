import hydra
from loguru import logger
from omegaconf import DictConfig

from tokenizer.artifacts import load_text, save_tokenizer
from tokenizer.bpe import BPETokenizer


@hydra.main(version_base=None, config_path="../config", config_name="train_tokenizer")
def main(cfg: DictConfig) -> None:
    text = load_text(cfg.data.input_path)

    tokenizer = BPETokenizer(special_tokens=list(cfg.tokenizer.special_tokens))
    tokenizer.train(text, vocab_size=cfg.tokenizer.vocab_size)

    tokenizer_path = save_tokenizer(
        tokenizer,
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )

    logger.info("Tokenizer saved to: {}", tokenizer_path)
    logger.info("Tokenizer vocab size: {}", tokenizer.vocab_size)


if __name__ == "__main__":
    main()
