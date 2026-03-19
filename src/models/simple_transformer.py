import math

import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleGPTPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, max_len, num_layers=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.max_len = max_len
        self.register_buffer("pe", self.positional_encoding(max_len, embed_size))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_size,
                num_heads,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=num_layers,
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens):
        positions = self.pe[: tokens.size(1), :]
        embedded = self.embedding(tokens) + positions
        causal_mask = self.generate_square_subsequent_mask(tokens.size(1), device=tokens.device)
        hidden = self.encoder(embedded, mask=causal_mask)
        return self.lm_head(hidden)

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def positional_encoding(self, max_len, embed_size):
        pe = torch.zeros(max_len, embed_size)
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_size)))
        return pe
