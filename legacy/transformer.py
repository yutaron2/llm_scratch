import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

device = "cuda:0"

# ============ CREATE VOCAB ONCE ============
with open('inputLearnText.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))  # SORTED for consistency!
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"Vocab size: {len(chars)}")
print("文字→ID辞書:", char_to_id)

def text_to_ids(text):
    return [char_to_id[ch] for ch in text]

def ids_to_text(ids):
    return ''.join([id_to_char[i] for i in ids])

# ============ MODEL DEFINITION ============
class SimpleGPTPredictor(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # token数の上限
        self.max_len = max_len
        # このへんでpos encoding
        self.pe = self.positional_encoding(max_len, embed_size) # (max_legnth, embed_size)


        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, batch_first=True),
            num_layers=2
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, batch_first=True),
            num_layers=2
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
        src_p = self.pe[:src.size(1), :].to(src.device) # (seq, embed_dim)
        target_p = self.pe[:src.size(1), :].to(tgt.device)

        # ソースをエンコード
        # batch_first=True なので (batch, seq, embed) のまま
        src_embedded = self.embedding(src) + src_p
        encoded = self.encoder(src_embedded) + target_p
        
        # ターゲットをデコード
        tgt_embedded = self.embedding(tgt)
        
        # ★追加3: 因果マスク (batch_first なので tgt.size(1) = seq_len)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))
        
        decoded = self.decoder(tgt_embedded, encoded, tgt_mask=tgt_mask)
        output = self.lm_head(decoded)
        
        return output
        
    # ★追加4: マスク生成メソッド
    def generate_square_subsequent_mask(self, sz):
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

# ============ DATA PREPARATION ============
def create_training_data(text, seq_len=10):
    ids = text_to_ids(text)
    src_data, tgt_data = [], []

    for i in range(len(ids) - seq_len):
        src_data.append(ids[i:i+seq_len])      # 入力：10文字
        tgt_data.append(ids[i+1:i+seq_len+1])  # 正解：1文字ずらした10文字

    return torch.tensor(src_data, device=device), torch.tensor(tgt_data, device=device)

train_src, train_tgt = create_training_data(text)

print(f"学習データ数: {len(train_src)}")
print(f"例 - 入力: '{ids_to_text(train_src[20].tolist())}'")
print(f"例 - 正解: '{ids_to_text(train_tgt[20].tolist())}'")

# ============ MODEL INITIALIZATION ============
model = SimpleGPTPredictor(vocab_size=len(chars), embed_size=32, num_heads=4, max_len=100)
model.to(device)

# 学習設定
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 損失関数の種類
criterion = nn.CrossEntropyLoss()

# ============ TRAINING LOOP ============
print("\n学習開始...")

for epoch in range(1000):
    total_loss = 0
    batch_size = 256
    
    for i in range(0, len(train_src), batch_size):
        optimizer.zero_grad()
        
        src_batch = train_src[i:i+batch_size]
        tgt_batch = train_tgt[i:i+batch_size]
        tgt_in = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]
        
        # ★変更: 2つの引数を渡す
        output = model(src_batch, tgt_in)
        
         # (batch, seq, vocab) ->  #(seq, )
        # 
        # ------- before -------
        # 
        # 
        # original_tensor = [
        # [  # バッチ1
        #     [1, 2, 3, 4],  # 位置1の予測ベクトル
        #     [5, 6, 7, 8],  # 位置2の予測ベクトル  
        #     [9,10,11,12]   # 位置3の予測ベクトル
        # ],
        # [  # バッチ2
        #     [13,14,15,16], # 位置1の予測ベクトル
        #     [17,18,19,20], # 位置2の予測ベクトル
        #     [21,22,23,24]  # 位置3の予測ベクトル
        # ]
        # ]
        #
        # ------- after -------
        #         reshaped_tensor = [
        #   [1, 2, 3, 4],   # バッチ1-位置1
        #   [5, 6, 7, 8],   # バッチ1-位置2  
        #   [9,10,11,12],   # バッチ1-位置3
        #   [13,14,15,16],  # バッチ2-位置1
        #   [17,18,19,20],  # バッチ2-位置2
        #   [21,22,23,24]   # バッチ2-位置3
        # ]
        # train_targetsは(batch*seq)の一次元。要するに次元を減らしている。（数値の変更とかはない）
        loss = criterion(output.reshape(-1, len(chars)), tgt_out.reshape(-1))
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch}, Loss: {total_loss/len(train_src):.4f}")
    
    save_dir = "model"
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{save_dir}/model_{epoch}.pth")

# ============ INFERENCE FUNCTIONS ============
def test_prediction(model: SimpleGPTPredictor, input_text):
    input_ids = text_to_ids(input_text)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model(input_tensor, input_tensor)  # Fixed: pass both src and tgt
        last_char_probs = output[0, -1, :]
        probs = torch.softmax(last_char_probs, dim=-1)

        # 一位のトークンを呼び出す。
        top_prob, top_index = torch.topk(probs, 1)
        char_id = top_index.item()  # テンソルから数値を取り出し
        predicted_char = id_to_char[char_id]  # IDを文字に変換

        return predicted_char

def generateSeq(model, text, count = 0):
    nextSingleToken = test_prediction(model, text)
    if(count < 20):
        return generateSeq(model, text+nextSingleToken, count+1)
    else:
        return text+nextSingleToken

# prompt = "The god is a"
# completion = generateSeq(model, prompt)
# print("入力テキスト: ", prompt)
# print("回答: ", completion)
