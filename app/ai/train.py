from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.training_data import TrainingExample, load_examples, write_corpus

DEFAULT_MODEL_CONFIG = {
    "block_size": 128,
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "dropout": 0.0,
}


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_instruction_sets(paths: list[Path]) -> tuple[list[TrainingExample], list[str]]:
    """Load multiple instruction sets and remove exact instruction/response duplicates."""
    merged: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()
    fingerprints: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        fingerprints.append(_file_fingerprint(path))
        for example in load_examples(path):
            key = (example.instruction.casefold(), example.response.casefold())
            if key in seen:
                continue
            seen.add(key)
            merged.append(example)
    return merged, fingerprints


def batchify(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too small for the configured block size")
    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


def evaluate(
    model: IndooneTransformer,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    batches: int,
    device: str,
) -> float | None:
    if batches <= 0:
        raise ValueError("batches must be greater than zero")
    if len(data) <= block_size + 1:
        return None
    model.eval()
    losses: list[float] = []
    generator = torch.Generator()
    generator.manual_seed(0)
    with torch.inference_mode():
        for _ in range(batches):
            x, y = batchify(data, block_size, batch_size, device, generator)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(float(loss.cpu()))
    model.train()
    return sum(losses) / len(losses)


def _snapshot_state(model: IndooneTransformer) -> dict[str, torch.Tensor]:
    """Clone model weights so a later optimizer step cannot mutate the snapshot."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train(
    corpus_path: Path,
    output_dir: Path,
    steps: int,
    seed: int,
    validation_path: Path | None = None,
    batch_size: int = 16,
    checkpoint_interval: int = 500,
    learning_rate: float = 3e-4,
    instruction_path: Path | None = None,
    multilingual_instruction_path: Path | None = None,
) -> float:
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be greater than zero")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_text = corpus_path.read_text(encoding="utf-8")
    instruction_paths = [
        path
        for path in (instruction_path, multilingual_instruction_path)
        if path is not None
    ]
    instruction_enabled = any(path.exists() for path in instruction_paths)
    instruction_fingerprints: list[str] = []
    instruction_example_count = 0
    if instruction_enabled:
        examples, instruction_fingerprints = _merge_instruction_sets(instruction_paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        instruction_corpus = output_dir / "instruction_corpus.txt"
        write_corpus(examples, instruction_corpus)
        train_text = train_text.rstrip() + "\n\n" + instruction_corpus.read_text(encoding="utf-8")
        instruction_example_count = len(examples)
    if len(train_text) < 32:
        raise ValueError("training corpus is too small; add more text")

    tokenizer = BPETokenizer.train(train_text, vocab_size=512, min_frequency=2)
    train_encoded = torch.tensor(
        tokenizer.encode(train_text, add_special_tokens=True),
        dtype=torch.long,
    )
    if len(train_encoded) < 4:
        raise ValueError("training corpus is too small after tokenization")

    block_size = min(128, max(2, len(train_encoded) // 2))
    if len(train_encoded) <= block_size + 1:
        raise ValueError("training corpus is too small for the selected block size")

    validation_encoded: torch.Tensor | None = None
    validation_fingerprint = None
    if validation_path is not None and validation_path.exists():
        validation_text = validation_path.read_text(encoding="utf-8")
        if validation_text.strip():
            validation_encoded = torch.tensor(
                tokenizer.encode(validation_text, add_special_tokens=True),
                dtype=torch.long,
            )
            validation_fingerprint = _file_fingerprint(validation_path)

    config = {**DEFAULT_MODEL_CONFIG, "block_size": block_size}
    model = IndooneTransformer(vocab_size=tokenizer.vocab_size, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    history: list[dict[str, float | int | None]] = []
    last_loss = float("inf")
    best_validation_loss = float("inf")
    best_validation_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None

    model.train()
    for step in range(1, steps + 1):
        x, y = batchify(train_encoded, block_size, batch_size, device, train_generator)
        _, loss = model(x, y)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

        if step == 1 or step % checkpoint_interval == 0 or step == steps:
            validation_loss = (
                evaluate(model, validation_encoded, block_size, batch_size, 4, device)
                if validation_encoded is not None
                else None
            )
            history.append(
                {
                    "step": step,
                    "train_loss": last_loss,
                    "validation_loss": validation_loss,
                }
            )

            if validation_loss is not None and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_step = step
                best_model_state = _snapshot_state(model)
                torch.save(
                    {
                        "step": step,
                        "model_config": model.config(),
                        "optimizer_state": optimizer.state_dict(),
                        "model_state": best_model_state,
                        "seed": seed,
                        "selection": "best_validation_loss",
                        "validation_loss": validation_loss,
                    },
                    output_dir / "best_checkpoint.pt",
                )

            torch.save(
                {
                    "step": step,
                    "model_config": model.config(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_state": model.state_dict(),
                    "seed": seed,
                },
                output_dir / "checkpoint.pt",
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    torch.save(
        {
            "config": config,
            "model_state": model.cpu().state_dict(),
        },
        output_dir / "indoone-small.pt",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": "indoone-small",
                "tokenizer": "bpe-v1",
                "vocab_size": tokenizer.vocab_size,
                "special_tokens": list(tokenizer.SPECIAL_TOKENS),
                "steps": steps,
                "seed": seed,
                "device": device,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "checkpoint_interval": checkpoint_interval,
                "validation_enabled": validation_encoded is not None,
                "best_validation_loss": (
                    best_validation_loss if best_model_state is not None else None
                ),
                "best_validation_step": best_validation_step,
                "final_model_selection": (
                    "best_validation_loss"
                    if best_model_state is not None
                    else "final_step"
                ),
                "instruction_data_enabled": instruction_enabled,
                "instruction_example_count": instruction_example_count,
                "instruction_fingerprints": instruction_fingerprints,
                "source_fingerprint": _file_fingerprint(corpus_path),
                "validation_fingerprint": validation_fingerprint,
                "training_text_characters": len(train_text),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return last_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the first Indoone local language model")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/train.txt"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.txt"))
    parser.add_argument("--instructions", type=Path, default=Path("data/raw/indoone_instructions.jsonl"))
    parser.add_argument("--multilingual-instructions", type=Path, default=Path("data/raw/indoone_multilingual_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/indoone-small"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    loss = train(
        args.corpus,
        args.output,
        args.steps,
        args.seed,
        args.validation,
        args.batch_size,
        args.checkpoint_interval,
        args.learning_rate,
        args.instructions,
        args.multilingual_instructions,
    )
    print(f"training complete; final loss={loss:.4f}")


if __name__ == "__main__":
    main()
