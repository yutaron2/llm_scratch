import pytest
import torch

from torch.utils.data import DataLoader

from src.datasets.text_dataset import (
    AutoregressiveTextDataset,
    create_autoregressive_dataloader,
    create_autoregressive_training_data,
)
from src.models.simple_decoder_transformer import SimpleDecoderTransformer


def test_create_autoregressive_training_data_uses_shifted_next_token_targets():
    inputs, labels = create_autoregressive_training_data(
        token_ids=[10, 11, 12, 13, 14],
        seq_len=3,
    )

    assert inputs.tolist() == [
        [10, 11, 12],
        [11, 12, 13],
    ]
    assert labels.tolist() == [
        [11, 12, 13],
        [12, 13, 14],
    ]


def test_create_autoregressive_training_data_emits_full_length_windows_without_bos_eos_insertion():
    bos_token_id = 101
    eos_token_id = 102

    inputs, labels = create_autoregressive_training_data(
        token_ids=[1, 2, 3, 4, 5],
        seq_len=2,
    )

    assert inputs.shape == (3, 2)
    assert labels.shape == (3, 2)
    assert bos_token_id not in inputs.tolist()
    assert bos_token_id not in labels.tolist()
    assert eos_token_id not in inputs.tolist()
    assert eos_token_id not in labels.tolist()


def test_create_autoregressive_training_data_requires_enough_tokens_for_labels():
    with pytest.raises(ValueError, match="Need at least 4 tokens"):
        create_autoregressive_training_data(
            token_ids=[10, 11, 12],
            seq_len=3,
        )


def test_autoregressive_text_dataset_and_dataloader_stream_windows():
    dataset = AutoregressiveTextDataset(
        token_ids=[10, 11, 12, 13, 14],
        seq_len=3,
    )

    assert len(dataset) == 2
    assert dataset[0]["inputs"].tolist() == [10, 11, 12]
    assert dataset[0]["labels"].tolist() == [11, 12, 13]

    loader = create_autoregressive_dataloader(
        token_ids=[10, 11, 12, 13, 14],
        seq_len=3,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert isinstance(loader, DataLoader)
    assert batch["inputs"].tolist() == [[10, 11, 12], [11, 12, 13]]
    assert batch["labels"].tolist() == [[11, 12, 13], [12, 13, 14]]


def test_simple_decoder_transformer_returns_vocab_logits():
    model = SimpleDecoderTransformer(
        vocab_size=32,
        embed_size=16,
        num_heads=4,
        max_len=4,
        num_layers=2,
    )
    tokens = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 0]])

    logits = model(tokens)

    assert logits.shape == (2, 4, 32)


def test_simple_decoder_transformer_builds_padding_mask_from_pad_token():
    model = SimpleDecoderTransformer(
        vocab_size=32,
        embed_size=16,
        num_heads=4,
        max_len=4,
        num_layers=1,
        dropout=0.0,
        pad_token_id=0,
    )
    tokens = torch.tensor([[1, 5, 0, 0], [2, 3, 4, 0]])

    padding_mask = model.make_padding_mask(tokens)

    assert padding_mask.tolist() == [
        [False, False, True, True],
        [False, False, False, True],
    ]


def test_decoder_block_forwards_key_padding_mask_to_attention():
    model = SimpleDecoderTransformer(
        vocab_size=32,
        embed_size=16,
        num_heads=4,
        max_len=4,
        num_layers=1,
        dropout=0.0,
        pad_token_id=0,
    )
    tokens = torch.tensor([[1, 5, 0, 0]])
    captured = {}

    original_forward = model.layers[0].self_attention.forward

    def wrapped_forward(*args, **kwargs):
        captured["key_padding_mask"] = kwargs.get("key_padding_mask")
        return original_forward(*args, **kwargs)

    model.layers[0].self_attention.forward = wrapped_forward
    try:
        model(tokens)
    finally:
        model.layers[0].self_attention.forward = original_forward

    assert captured["key_padding_mask"].tolist() == [[False, False, True, True]]
