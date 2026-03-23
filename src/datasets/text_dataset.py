import torch
from torch.utils.data import DataLoader, Dataset


class AutoregressiveTextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        if len(token_ids) < seq_len + 1:
            raise ValueError(
                f"Not enough tokenized data for seq_len={seq_len}. "
                f"Need at least {seq_len + 1} tokens, but got {len(token_ids)}."
            )

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.token_ids) - self.seq_len

    def __getitem__(self, index):
        start = index
        end = index + self.seq_len
        return {
            "inputs": self.token_ids[start:end],
            "labels": self.token_ids[start + 1 : end + 1],
        }


def create_autoregressive_training_data(token_ids, seq_len, device=None):
    dataset = AutoregressiveTextDataset(token_ids=token_ids, seq_len=seq_len)
    inputs = []
    labels = []

    for index in range(len(dataset)):
        sample = dataset[index]
        inputs.append(sample["inputs"])
        labels.append(sample["labels"])

    return (
        torch.stack(inputs).to(device=device),
        torch.stack(labels).to(device=device),
    )


def create_autoregressive_dataloader(token_ids, seq_len, batch_size, shuffle=True):
    dataset = AutoregressiveTextDataset(token_ids=token_ids, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
