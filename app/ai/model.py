from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODEL_VERSION = "indoone-gpt-v2"


class DecoderBlock(nn.Module):
    """Pre-norm causal Transformer block used by the local Indoone model."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        email_adapter_dim: int = 0,
    ) -> None:
        super().__init__()
        if email_adapter_dim < 0:
            raise ValueError("email_adapter_dim must be non-negative")
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
        self.email_adapter_dim = email_adapter_dim
        if email_adapter_dim:
            self.email_adapter_down = nn.Linear(n_embd, email_adapter_dim, bias=False)
            self.email_adapter_up = nn.Linear(email_adapter_dim, n_embd, bias=False)
            nn.init.normal_(self.email_adapter_down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.email_adapter_up.weight)
        else:
            self.email_adapter_down = None
            self.email_adapter_up = None

    def _apply_email_adapter(
        self,
        x: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled or self.email_adapter_down is None or self.email_adapter_up is None:
            return x
        adapter_input = self.ln_2(x)
        adapter_hidden = F.gelu(self.email_adapter_down(adapter_input))
        return x + self.email_adapter_up(adapter_hidden)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        use_email_adapter: bool = False,
    ) -> torch.Tensor:
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
        x = x + self.mlp(self.ln_2(x))
        return self._apply_email_adapter(x, use_email_adapter)

    @torch.no_grad()
    def forward_cached(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_email_adapter: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run this block while reusing cached attention keys and values."""
        if x.ndim != 3:
            raise ValueError("cached block input must have shape (batch, sequence, embedding)")

        normalized = self.ln_1(x)
        batch_size, sequence_length, embedding_size = normalized.shape
        expected_size = self.attn.embed_dim
        if embedding_size != expected_size:
            raise ValueError("cached block input embedding size does not match attention")

        qkv = F.linear(
            normalized,
            self.attn.in_proj_weight,
            self.attn.in_proj_bias,
        )
        query, key, value = qkv.chunk(3, dim=-1)

        head_dim = self.attn.head_dim
        num_heads = self.attn.num_heads

        def reshape_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                num_heads,
                head_dim,
            ).transpose(1, 2)

        query = reshape_heads(query)
        key = reshape_heads(key)
        value = reshape_heads(value)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            if past_key.ndim != 4 or past_value.ndim != 4:
                raise ValueError("cached attention state must contain 4D key/value tensors")
            if past_key.shape[0] != batch_size or past_value.shape[0] != batch_size:
                raise ValueError("cached attention state batch size does not match input")
            if past_key.shape[1] != num_heads or past_value.shape[1] != num_heads:
                raise ValueError("cached attention state head count does not match model")
            if past_key.shape[-1] != head_dim or past_value.shape[-1] != head_dim:
                raise ValueError("cached attention state head size does not match model")

            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
            causal = False
        else:
            causal = True

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            expected_size,
        )
        attention = self.attn.out_proj(attention)

        x = x + attention
        x = x + self.mlp(self.ln_2(x))
        x = self._apply_email_adapter(x, use_email_adapter)
        return x, (key, value)


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
        email_adapter_dim: int = 0,
        email_adapter_layers: int = 0,
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
        if email_adapter_dim < 0:
            raise ValueError("email_adapter_dim must be non-negative")
        if email_adapter_layers < 0 or email_adapter_layers > n_layer:
            raise ValueError("email_adapter_layers must be between 0 and n_layer")
        if email_adapter_layers and not email_adapter_dim:
            raise ValueError("email_adapter_dim must be positive when email_adapter_layers is enabled")

        self.model_version = MODEL_VERSION
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.email_adapter_dim = email_adapter_dim
        self.email_adapter_layers = email_adapter_layers

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        adapter_start = n_layer - email_adapter_layers
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    n_embd,
                    n_head,
                    dropout,
                    email_adapter_dim=email_adapter_dim if index >= adapter_start else 0,
                )
                for index in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        # The Email adapter must start as an exact identity so enabling it on
        # an untrained adapter cannot perturb the existing base model.
        for block in self.blocks:
            if block.email_adapter_up is not None:
                nn.init.zeros_(block.email_adapter_up.weight)

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
            "email_adapter_dim": self.email_adapter_dim,
            "email_adapter_layers": self.email_adapter_layers,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_email_adapter: bool = False,
    ):
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
            x = block(
                x,
                causal_mask,
                use_email_adapter=use_email_adapter,
            )

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

    @torch.inference_mode()
    def forward_cached(
        self,
        idx: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_email_adapter: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Run one prompt segment or generated tokens using KV caching."""
        if idx.ndim != 2:
            raise ValueError("cached input tensor must have shape (batch, sequence)")
        batch_size, length = idx.shape
        if length <= 0:
            raise ValueError("cached input sequence must not be empty")

        if past_key_values is None:
            past_length = 0
            normalized_cache = tuple(None for _ in self.blocks)
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("cached layer count does not match model")
            if not past_key_values:
                raise ValueError("cached layer state cannot be empty")
            if length != 1:
                raise ValueError("cached decoding accepts one generated token at a time")
            first_key = past_key_values[0][0]
            past_length = int(first_key.size(2))
            normalized_cache = tuple(past_key_values)

        if past_length + length > self.block_size:
            raise ValueError("cached input sequence exceeds model block size")

        positions = torch.arange(
            past_length,
            past_length + length,
            device=idx.device,
        ).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)

        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block, layer_cache in zip(self.blocks, normalized_cache):
            x, layer_cache = block.forward_cached(
                x,
                layer_cache,
                use_email_adapter=use_email_adapter,
            )
            new_cache.append(layer_cache)

        logits = self.lm_head(self.ln_f(x))
        return logits, tuple(new_cache)
