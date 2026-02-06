import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ MODEL DEFINITION ============
class SimpleGPTPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, max_len, num_layers=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # token数の上限
        self.max_len = max_len
        # このへんでpos encoding
        # Register as buffer so it moves with .to(device)
        self.register_buffer("pe", self.positional_encoding(max_len, embed_size))


        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, batch_first=True),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, batch_first=True),
            num_layers=num_layers
        )

        self.lm_head = nn.Linear(embed_size, vocab_size)


    def forward(self, src, tgt): # (b, seq_length)
        """
        pe =
        [
            [12, 123, 1, 4, 5 ,1 ,4],
            [12, 123, 1, 4, 5 ,1 ,4]
        ]
        [
            [13434, 2343,  3234,...],
            [32432, 343324, 4343,...],
            [],
        ]
        """
        # scr（peは↑のような固定値の配列なので、入力エンべディングのサイズに合わせて切り取る）
        src_p = self.pe[:src.size(1), :]  # (seq, embed_dim)
        tgt_p = self.pe[:tgt.size(1), :]

        # ソースをエンコード
        # batch_first=True なので (batch, seq, embed) のまま
        src_embedded = self.embedding(src) + src_p
        encoded = self.encoder(src_embedded)

        # ターゲットをデコード
        tgt_embedded = self.embedding(tgt) + tgt_p

        # ★追加3: 因果マスク (batch_first なので tgt.size(1) = seq_len)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)

        decoded = self.decoder(tgt_embedded, encoded, tgt_mask=tgt_mask)
        output = self.lm_head(decoded)

        return output

    # ★追加4: マスク生成メソッド
    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def positional_encoding(self, max_len, embed_size):
        pe = torch.zeros(max_len, embed_size) # [max_length, embedd_size]
        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                pe[pos, i]     = math.sin(pos / (10000 ** (i / embed_size)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_size)))
        return pe
