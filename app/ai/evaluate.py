from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")


def evaluate_checkpoint(
    checkpoint_path: Path,
    tokenizer_path: Path,
    corpus_path: Path,
    batch_size: int = 16,
) -> dict[str, float | int | str]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    tokenizer = BPETokenizer.load(tokenizer_path)
    config = checkpoint["config"]
    model = IndooneTransformer(vocab_size=tokenizer.vocab_size, **config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text = corpus_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("evaluation corpus cannot be empty")

    encoded = torch.tensor(
        tokenizer.encode(text, add_special_tokens=True),
        dtype=torch.long,
    )
    block_size = model.block_size
    if len(encoded) <= block_size + 1:
        raise ValueError("evaluation corpus is too small for the model block size")

    starts = range(0, len(encoded) - block_size, block_size)
    losses: list[float] = []
    examples = 0

    with torch.inference_mode():
        for offset in range(0, len(list(starts)), batch_size):
            chunk = list(starts)[offset : offset + batch_size]
            if not chunk:
                continue
            x = torch.stack([encoded[i : i + block_size] for i in chunk])
            y = torch.stack([encoded[i + 1 : i + block_size + 1] for i in chunk])
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss))
            examples += len(chunk)

    loss = sum(losses) / len(losses)
    return {
        "model_version": str(config["model_version"]),
        "corpus_tokens": int(len(encoded)),
        "evaluation_batches": examples,
        "loss": loss,
        "perplexity": math.exp(min(loss, 20.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Indoone local language model")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        corpus_path=args.corpus,
        batch_size=args.batch_size,
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
