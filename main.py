import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from model import SimpleGPTPredictor, device
from train_tokenizer import BPETokenizer


INPUT_PATH = "inputLearnText.txt"
TOKENIZER_VOCAB_SIZE = 512
SEQ_LEN = 100
SAVE_ROOT = "model"


with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size=TOKENIZER_VOCAB_SIZE)

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
if tokenizer.merges:
    print(f"最初のマージ: {tokenizer.describe_merge(0)}")


def text_to_ids(text):
    return tokenizer.encode(text)


def ids_to_text(ids):
    return tokenizer.decode(ids)


def create_training_data(text, seq_len=SEQ_LEN):
    ids = text_to_ids(text)
    input_data, target_data = [], []

    for i in range(len(ids) - seq_len):
        window = ids[i:i + seq_len + 1]
        input_data.append(window[:-1])
        target_data.append(window[1:])

    if not input_data:
        raise ValueError(
            f"Not enough tokenized data for seq_len={seq_len}. "
            f"Corpus produced only {len(ids)} tokens."
        )

    return torch.tensor(input_data, device=device), torch.tensor(target_data, device=device)


train_inputs, train_targets = create_training_data(text)
sample_index = min(20, len(train_inputs) - 1)

print(f"学習データ数: {len(train_inputs)}")
print(f"例 - 入力: '{ids_to_text(train_inputs[sample_index].tolist())}'")
print(f"例 - 正解: '{ids_to_text(train_targets[sample_index].tolist())}'")


model = SimpleGPTPredictor(
    vocab_size=tokenizer.vocab_size,
    embed_size=512,
    num_heads=4,
    max_len=SEQ_LEN,
)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
criterion = nn.CrossEntropyLoss()

print("\n学習開始...")

for epoch in range(1000):
    total_loss = 0
    batch_size = 32
    num_batches = math.ceil(len(train_inputs) / batch_size)

    for i in tqdm(range(0, len(train_inputs), batch_size)):
        optimizer.zero_grad()

        input_batch = train_inputs[i:i + batch_size]
        target_batch = train_targets[i:i + batch_size]

        output = model(input_batch, input_batch)
        loss = criterion(output.reshape(-1, tokenizer.vocab_size), target_batch.reshape(-1))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch}, Loss: {avg_loss:.6f}, LR: {current_lr:.6e}")
    scheduler.step()

    version_dir = os.path.join(SAVE_ROOT, f"model_{epoch}")
    os.makedirs(version_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(version_dir, "model.pth"))
    tokenizer.save(os.path.join(version_dir, "tokenizer.json"))


def test_prediction(model: SimpleGPTPredictor, input_text, temperature=0.0):
    input_ids = text_to_ids(input_text)
    if len(input_ids) < 1:
        raise ValueError("At least 1 token is required for inference with the current model.")

    if len(input_ids) > SEQ_LEN:
        input_ids = input_ids[-SEQ_LEN:]

    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model(input_tensor, input_tensor)
        last_token_probs = output[0, -1, :]

        if temperature <= 0:
            predicted_token_id = int(torch.argmax(last_token_probs).item())
        else:
            probs = torch.softmax(last_token_probs / temperature, dim=-1)
            predicted_token_id = int(torch.multinomial(probs, num_samples=1).item())

        predicted_piece = ids_to_text([predicted_token_id])

        return predicted_piece


def generateSeq(model, text, count=0, temperature=0.0):
    next_token = test_prediction(model, text, temperature=temperature)
    if count < 20:
        return generateSeq(model, text + next_token, count + 1, temperature=temperature)
    return text + next_token


# prompt = "The god is a"
# completion = generateSeq(model, prompt)
# print("入力テキスト: ", prompt)
# print("回答: ", completion)
