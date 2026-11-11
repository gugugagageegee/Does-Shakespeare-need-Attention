import torch


def split_data(
    token_ids,
    train_ratio=0.9,
    val_ratio=0.05
):
    token_ids = torch.as_tensor(
        token_ids,
        dtype=torch.long
    )

    total_length = len(token_ids)

    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("Split ratios must be positive.")

    if train_ratio + val_ratio >= 1:
        raise ValueError(
            "Train ratio + validation ratio must be below 1."
        )

    train_end = int(total_length * train_ratio)

    val_end = int(
        total_length * (train_ratio + val_ratio)
    )

    train_data = token_ids[:train_end]

    val_data = token_ids[train_end:val_end]

    test_data = token_ids[val_end:]

    return train_data, val_data, test_data


def get_batch(
    data,
    batch_size,
    block_size,
    device
):
    if len(data) <= block_size:
        raise ValueError(
            "Dataset must contain more tokens than block_size."
        )

    # Random starting positions
    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    offsets = torch.arange(block_size)

    # Input sequences
    x = data[starts[:, None] + offsets]

    # Next-token targets
    y = data[starts[:, None] + offsets + 1]

    return (
        x.to(device),
        y.to(device)
    )