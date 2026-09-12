from __future__ import annotations

import torch
from torch import nn


MODEL_VERSION = "indoone-gpt-v2"


class DecoderBlock(nn.Module):
    """Pre-norm causal Transformer block used by the local Indoone model."""

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        normalized = self.ln_1(x)
        attention, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
            is_causal=True,
        )
        x = x + attention
        return x + self.mlp(self.ln_2(x))


class IndooneTransformer(nn.Module):
    """Configurable decoder-only Transformer for Indoone local inference."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        n_embd: int = 384,
        n_head: int = 8,
        n_layer: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero")
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        if n_embd <= 0:
            raise ValueError("n_embd must be greater than zero")
        if n_head <= 0:
            raise ValueError("n_head must be greater than zero")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if n_layer <= 0:
            raise ValueError("n_layer must be greater than zero")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.model_version = MODEL_VERSION
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def config(self) -> dict[str, int | float | str]:
        return {
            "model_version": self.model_version,
            "block_size": self.block_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "dropout": self.dropout,
        }

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if idx.ndim != 2:
            raise ValueError("input tensor must have shape (batch, sequence)")
        _, length = idx.shape
        if length <= 0:
            raise ValueError("input sequence must not be empty")
        if length > self.block_size:
            raise ValueError("input sequence exceeds model block size")

        positions = torch.arange(length, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)
        causal_mask = torch.triu(
            torch.ones(length, length, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must match input tensor shape")
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss
