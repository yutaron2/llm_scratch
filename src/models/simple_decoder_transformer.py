import torch
import torch.nn as nn

from models.embedding import TransformerEmbedding


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, dim_feedforward):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_size),
        )
        self.feedforward_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, causal_mask):
        attention_output, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + self.dropout(attention_output))
        feedforward_output = self.feedforward(hidden)
        return self.feedforward_norm(hidden + self.dropout(feedforward_output))


class SimpleDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        max_len,
        num_layers=4,
        dropout=0.1,
        dim_feedforward=None,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = embed_size * 4

        self.max_len = max_len
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            max_len=max_len,
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens):
        sequence_length = tokens.size(1)
        if sequence_length > self.max_len:
            raise ValueError(
                f"Sequence length exceeds max_len={self.max_len}: {sequence_length}"
            )

        hidden = self.embedding(tokens)
        causal_mask = self.generate_square_subsequent_mask(
            sequence_length,
            device=tokens.device,
        )
        for layer in self.layers:
            hidden = layer(hidden, causal_mask=causal_mask)
        return self.lm_head(hidden)

    @staticmethod
    def generate_square_subsequent_mask(size, device):
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
