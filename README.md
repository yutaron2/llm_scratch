# llm_scratch

A small scratch project for experimenting with a decoder-only autoregressive Transformer and a character-level BPE tokenizer.

## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) installed
- Python 3.10+

### Create or update the environment
```bash
make sync
```

This uses `uv sync --dev` to create or update the local `.venv` from `pyproject.toml`, and `uv` will generate or refresh `uv.lock` as needed.

### Activate the environment
```bash
make activate
```

`make activate` prints the command you should run in your current shell:

```bash
source .venv/bin/activate
```

A Make target cannot directly modify the parent shell, so the activation command must be run manually.

## Common commands

### Run tests
```bash
make test
```

### Train the tokenizer
```bash
uv run python src/train_tokenizer.py
```

This produces the tokenizer artifact configured by `artifacts.tokenizers_dir` and `artifacts.tokenizer_filename`.

### Run the model training script
```bash
make run
```

This launches the Hydra-based training entrypoint, which expects an already-trained tokenizer artifact:

```bash
uv run python src/train.py
```

Training uses a decoder-only autoregressive setup built from one corpus:
- each sample is a left-to-right language modeling window served lazily from a `Dataset`/`DataLoader` pipeline
- the tokenized corpus is treated as one continuous stream
- each training input is a contiguous slice of that stream with fixed length
- labels are the next-token-shifted slice for standard causal language modeling

At inference time, the model predicts one tokenizer token at a time, not one whole word at a time. Because the tokenizer is character-level BPE, a word like `give` may be produced over multiple decoding steps such as `g`, `iv`, then `e `.

You can override runtime values with Hydra arguments, for example:

```bash
uv run python src/train.py training.epochs=10 training.batch_size=64
```

## Project files
- `src/train.py`: Hydra-based decoder-only training script that loads a saved tokenizer
- `src/train_tokenizer.py`: Hydra-based tokenizer training script
- `src/models/embedding.py`: token embedding and sinusoidal positional encoding
- `src/models/simple_decoder_transformer.py`: GPT-style decoder-only Transformer blocks with causal self-attention
- `src/tokenizer/bpe.py`: character-level BPE tokenizer implementation
- `src/datasets/text_dataset.py`: decoder-only autoregressive dataset and dataloader helpers
- `config/train.yaml`: model training configuration
- `config/train_tokenizer.yaml`: tokenizer training configuration
- `data/inputLearnText.txt`: training corpus
- `tests/test_decoder_only.py`: decoder-only data/model smoke tests

## References
- 「[大規模言語モデル入門](https://www.amazon.co.jp/%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%E5%85%A5%E9%96%80-%E5%B1%B1%E7%94%B0-%E8%82%B2%E7%9F%A2/dp/4297136333/ref=asc_df_4297136333?mcid=250d2916cf3a37869b7de666d8c23fb5&th=1&psc=1&tag=jpgo-22&linkCode=df0&hvadid=707442440817&hvpos=&hvnetw=g&hvrand=8178253711343895759&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1009213&hvtargid=pla-2198420664173&psc=1&hvocijid=8178253711343895759-4297136333-&hvexpln=0)」
- 「[ゼロから作るDeep Learning](https://www.oreilly.co.jp/books/9784873117584/)」
- 「[ゼロから作るDeep Learning❷ ―自然言語処理編](https://www.amazon.co.jp/%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8BDeep-Learning-%E2%80%95%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E7%B7%A8-%E6%96%8E%E8%97%A4-%E5%BA%B7%E6%AF%85/dp/4873118360/ref=sr_1_1?crid=2TYPO4NHHKAI&dib=eyJ2IjoiMSJ9.8yXWKZwPXXXhKJLL2lNnbt32KW2y8fK6pshjKzHRuZ1fSfGgASV3i6W6v9mO_I658nDbHiaxiEda-cE7ROIRBg.rw9pDRx17ZWOyzR8dhTIy4NEDLuY6xr03krPvys0oC0&dib_tag=se&keywords=%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8Bdeep+learning+%E8%87%AA%E7%84%B6%E8%A8%80%E8%AA%9E%E5%87%A6%E7%90%86%E7%B7%A8&qid=1768661759&sprefix=%E3%82%BC%E3%83%AD%E3%81%8B%E3%82%89%E4%BD%9C%E3%82%8B%2Caps%2C208&sr=8-1)」
- "[Attention is All You Need](https://arxiv.org/abs/1706.03762)"
