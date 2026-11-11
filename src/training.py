from pathlib import Path
from dataclasses import asdict

import torch

from .dataset import get_batch
from .evaluation import estimate_loss


def train_model(
    model,
    train_data,
    val_data,
    max_iters=2000,
    batch_size=32,
    block_size=64,
    learning_rate=3e-4,
    weight_decay=0.01,
    eval_interval=100,
    eval_iters=50,
    grad_clip=1.0,
    device="cpu"
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    history = []

    best_val_loss = float("inf")
    best_state = None

    for step in range(1, max_iters + 1):

        model.train()

        # Get training batch
        x, y = get_batch(
            train_data,
            batch_size,
            block_size,
            device
        )

        # Forward pass
        _, loss = model(x, y)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip
        )

        # Update parameters
        optimizer.step()

        # Periodic evaluation
        if (
            step % eval_interval == 0
            or step == 1
            or step == max_iters
        ):

            metrics = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=batch_size,
                block_size=block_size,
                eval_iters=eval_iters,
                device=device
            )

            train_loss = metrics["train"]
            val_loss = metrics["val"]

            history.append({
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss
            })

            print(
                f"Step {step:5d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Keep the best validation checkpoint in memory
            if val_loss < best_val_loss:

                best_val_loss = val_loss

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    # Restore the best validation model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation loss: {best_val_loss:.4f}")

    return history


def save_checkpoint(
    path,
    model,
    tokenizer,
    step=None,
    val_loss=None
):

    if not hasattr(model, "config"):
        raise ValueError(
            "save_checkpoint expects a MiniGPT model."
        )

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },

        "model_config": asdict(model.config),

        "tokenizer": tokenizer.to_dict(),

        "step": step,

        "val_loss": val_loss
    }

    torch.save(checkpoint, path)

    print(f"Checkpoint saved: {path}")