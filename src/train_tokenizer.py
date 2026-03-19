import hydra
from omegaconf import DictConfig

from tokenizer.artifacts import load_text, save_tokenizer
from tokenizer.bpe import BPETokenizer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    text = load_text(cfg.data.input_path)

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=cfg.tokenizer.vocab_size)

    tokenizer_path = save_tokenizer(
        tokenizer,
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )

    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    if tokenizer.merges:
        print(f"最初のマージ: {tokenizer.describe_merge(0)}")


if __name__ == "__main__":
    main()
