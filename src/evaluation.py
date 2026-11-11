import math

import torch

from .dataset import get_batch


@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    batch_size,
    block_size,
    eval_iters,
    device
):

    was_training = model.training

    model.eval()

    results = {}

    splits = {
        "train": train_data,
        "val": val_data
    }

    for split_name, data in splits.items():

        losses = []

        for _ in range(eval_iters):

            x, y = get_batch(
                data,
                batch_size,
                block_size,
                device
            )

            _, loss = model(x, y)

            losses.append(loss.item())

        results[split_name] = sum(losses) / len(losses)

    model.train(was_training)

    return results


def calculate_perplexity(loss):

    if loss < 0:
        raise ValueError("Loss must be non-negative.")

    return math.exp(loss)