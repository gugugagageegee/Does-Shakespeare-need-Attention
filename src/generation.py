from pathlib import Path

import torch

from .model import GPTConfig, MiniGPT
from .tokenizer import CharTokenizer


def load_checkpoint(
    path,
    device="cpu"
):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    # Load checkpoint
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            path,
            map_location=device
        )

    # Restore model configuration
    config = GPTConfig(
        **checkpoint["model_config"]
    )

    model = MiniGPT(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    # Restore tokenizer
    tokenizer = CharTokenizer.from_dict(
        checkpoint["tokenizer"]
    )

    return model, tokenizer, checkpoint

@torch.no_grad()
def generate(
    model,
    idx,
    max_new_tokens,
    block_size,
    temperature=1.0,
    top_k=None
):

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be non-negative."
        )

    if top_k is not None and top_k <= 0:
        raise ValueError(
            "top_k must be positive."
        )

    was_training = model.training

    model.eval()

    try:

        for _ in range(max_new_tokens):

            # Keep only the latest context
            idx_cond = idx[:, -block_size:]

            # Predict next-token logits
            logits, _ = model(idx_cond)

            logits = logits[:, -1, :]

            # Temperature scaling
            logits = logits / temperature

            # Optional Top-K sampling
            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                threshold = values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            # Convert logits to probabilities
            probs = torch.softmax(
                logits,
                dim=-1
            )

            # Sample next token
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Append new token
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

    finally:
        model.train(was_training)

    return idx