import torch


def create_training_data(token_ids, seq_len, device=None):
    input_data = []
    target_data = []

    for index in range(len(token_ids) - seq_len):
        window = token_ids[index : index + seq_len + 1]
        input_data.append(window[:-1])
        target_data.append(window[1:])

    if not input_data:
        raise ValueError(
            f"Not enough tokenized data for seq_len={seq_len}. "
            f"Corpus produced only {len(token_ids)} tokens."
        )

    return (
        torch.tensor(input_data, device=device, dtype=torch.long),
        torch.tensor(target_data, device=device, dtype=torch.long),
    )
