import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.embed_size)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_size)
        )

        encoding = torch.zeros(max_len, embed_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])

        self.register_buffer("encoding", encoding)

    def forward(self, tokens):
        seq_len = tokens.size(1)
        return self.encoding[:seq_len].unsqueeze(0)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len):
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embed_size)
        self.position = SinusoidalPositionalEncoding(max_len, embed_size)

    def forward(self, tokens):
        return self.token(tokens) + self.position(tokens)
