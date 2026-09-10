from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import CharacterTokenizer


def batchify(data: torch.Tensor, block_size: int, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


def train(corpus_path: Path, output_dir: Path, steps: int, seed: int) -> float:
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = corpus_path.read_text(encoding="utf-8")
    if len(text) < 32:
        raise ValueError("training corpus is too small; add more text")

    tokenizer = CharacterTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    config = {
        "block_size": min(128, max(32, len(encoded) // 4)),
        "n_embd": 128,
        "n_head": 4,
        "n_layer": 4,
        "dropout": 0.0,
    }
    model = IndooneTransformer(vocab_size=len(tokenizer.chars), **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    model.train()
    last_loss = float("inf")
    for _ in range(steps):
        x, y = batchify(encoded, config["block_size"], 16, device)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    torch.save(
        {"config": config, "model_state": model.cpu().state_dict()},
        output_dir / "indoone-small.pt",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps({"model": "indoone-small", "steps": steps, "seed": seed, "device": device}, indent=2),
        encoding="utf-8",
    )
    return last_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first Indoone local language model")
    parser.add_argument("--corpus", type=Path, default=Path("data/indoone_corpus.txt"))
    parser.add_argument("--output", type=Path, default=Path("models/indoone-small"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    loss = train(args.corpus, args.output, args.steps, args.seed)
    print(f"training complete; final loss={loss:.4f}")


if __name__ == "__main__":
    main()
