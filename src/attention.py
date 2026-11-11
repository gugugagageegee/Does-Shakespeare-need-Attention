import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q,
    k,
    v,
    mask=None,
    dropout=None
):
    # Attention scores
    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    )

    scores = scores / math.sqrt(q.size(-1))

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min
        )

    # Attention weights
    weights = F.softmax(
        scores,
        dim=-1
    )

    # Attention dropout
    if dropout is not None:
        weights = dropout(weights)

    # Weighted sum of values
    output = torch.matmul(weights, v)

    return output, weights

class CausalSelfAttentionHead(nn.Module):

    def __init__(
        self,
        n_embd,
        head_size,
        block_size,
        dropout
    ):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.attn_dropout = nn.Dropout(dropout)

        # Lower triangular causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size,
                    dtype=torch.bool
                )
            )
        )

    def forward(self, x):

        B, T, C = x.shape

        if T > self.causal_mask.size(0):
            raise ValueError("Sequence exceeds block_size.")

        q = self.query(x)

        k = self.key(x)

        v = self.value(x)

        mask = self.causal_mask[:T, :T]

        output, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=self.attn_dropout
        )

        return output, weights

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        n_heads,
        block_size,
        dropout
    ):
        super().__init__()

        if n_embd % n_heads != 0:
            raise ValueError(
                "n_embd must be divisible by n_heads."
            )

        head_size = n_embd // n_heads

        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                n_embd,
                head_size,
                block_size,
                dropout
            )
            for _ in range(n_heads)
        ])

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):

        outputs = []
        attention_weights = []

        for head in self.heads:

            head_output, weights = head(x)

            outputs.append(head_output)

            attention_weights.append(weights)

        # Concatenate all heads
        out = torch.cat(
            outputs,
            dim=-1
        )

        # Final linear projection
        out = self.proj(out)

        out = self.resid_dropout(out)

        if return_attention:
            attention_weights = torch.stack(
                attention_weights,
                dim=1
            )

            return out, attention_weights

        return out

