from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import CharacterTokenizer


def batchify(data: torch.Tensor, block_size: int, batch_size: int):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the small local Indoone language model."
    )
    parser.add_argument("--data", type=Path, default=Path("data/indoone_corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("models/indoone-small.pt"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("models/indoone-tokenizer.json")
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    text = args.data.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    encoded = tokenizer.encode(text)
    if len(encoded) <= args.block_size + 1:
        raise ValueError("corpus is too small for configured block size")
    data = torch.tensor(encoded, dtype=torch.long)

    config = {
        "block_size": args.block_size,
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.0,
    }
    model = IndooneTransformer(vocab_size=len(tokenizer.chars), **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    for step in range(args.steps):
        x, y = batchify(data, args.block_size, args.batch_size)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0:
            print(f"step={step + 1} loss={loss.item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer)
    torch.save({"config": config, "model_state": model.state_dict()}, args.output)
    print(f"saved model: {args.output}")
    print(f"saved tokenizer: {args.tokenizer}")


if __name__ == "__main__":
    main()
