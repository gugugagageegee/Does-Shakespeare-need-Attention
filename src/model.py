from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import TransformerBlock


@dataclass
class GPTConfig:

    vocab_size: int

    block_size: int = 64

    n_embd: int = 128

    n_heads: int = 4

    n_layers: int = 2

    ffn_dim: int = 512

    dropout: float = 0.1

    def __post_init__(self):

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.n_embd % self.n_heads != 0:
            raise ValueError(
                "n_embd must be divisible by n_heads."
            )

        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

class MiniGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token Embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # Positional Embedding
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        # Stack Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                n_embd=config.n_embd,
                n_heads=config.n_heads,
                ffn_dim=config.ffn_dim,
                block_size=config.block_size,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(
            config.n_embd
        )

        # Language Model Head
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(
        self,
        idx,
        targets=None,
        return_attention=False
    ):

        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                "Input sequence is longer than block_size."
            )

        # Token and position embeddings
        token_emb = self.token_embedding(idx)

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb.unsqueeze(0)

        x = self.embedding_dropout(x)

        # Transformer Blocks
        attention_maps = []

        for block in self.blocks:

            if return_attention:

                x, weights = block(
                    x,
                    return_attention=True
                )

                attention_maps.append(weights)

            else:
                x = block(x)

        # Final representation
        x = self.final_ln(x)

        # Next-character logits
        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        if return_attention:
            return logits, loss, attention_maps

        return logits, loss

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )