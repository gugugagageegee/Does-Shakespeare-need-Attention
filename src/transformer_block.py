import torch.nn as nn

from .attention import MultiHeadSelfAttention


class FeedForward(nn.Module):

    def __init__(
        self,
        n_embd,
        ffn_dim,
        dropout
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, ffn_dim),

            nn.GELU(),

            nn.Linear(ffn_dim, n_embd),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(
        self,
        n_embd,
        n_heads,
        ffn_dim,
        block_size,
        dropout
    ):
        super().__init__()

        # Pre-LayerNorm
        self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

        # Multi-Head Self-Attention
        self.attn = MultiHeadSelfAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout
        )

        # Feed Forward Network
        self.ffn = FeedForward(
            n_embd=n_embd,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

    def forward(self, x, return_attention=False):

        # Attention branch
        if return_attention:
            attn_out, attention_weights = self.attn(
                self.ln1(x),
                return_attention=True
            )
        else:
            attn_out = self.attn(
                self.ln1(x)
            )

        # First residual connection
        x = x + attn_out

        # Feed Forward branch
        ffn_out = self.ffn(
            self.ln2(x)
        )

        # Second residual connection
        x = x + ffn_out

        if return_attention:
            return x, attention_weights

        return x