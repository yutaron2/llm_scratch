from pathlib import Path

import hydra
from omegaconf import DictConfig

from tokenizer.bpe import BPETokenizer


ROOT_DIR = Path(__file__).resolve().parent.parent


def load_text(path: str) -> str:
    return (ROOT_DIR / path).read_text(encoding="utf-8")


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    text = load_text(cfg.data.input_path)

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=cfg.tokenizer.vocab_size)

    tokenizer_path = ROOT_DIR / cfg.artifacts.tokenizers_dir / cfg.artifacts.tokenizer_filename
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))

    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    if tokenizer.merges:
        print(f"最初のマージ: {tokenizer.describe_merge(0)}")


if __name__ == "__main__":
    main()
