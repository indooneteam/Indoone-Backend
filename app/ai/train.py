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


# V2 is intentionally larger while remaining practical for CPU inference.
DEFAULT_MODEL_CONFIG = {
    "block_size": 512,
    "n_embd": 384,
    "n_head": 8,
    "n_layer": 10,
    "dropout": 0.0,
}
DEFAULT_VOCAB_SIZE = 8192
DEFAULT_MIN_FREQUENCY = 2


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


def _merge_instruction_sets_weighted(
    paths: list[Path],
) -> tuple[list[TrainingExample], list[str], list[float], dict[str, dict[str, int | float]]]:
    """Merge instruction pools with explicit sampling priority.

    Curated behavior examples are deliberately oversampled relative to synthetic
    augmentation so the evaluation-driving examples are not drowned out by a
    much larger generated pool.
    """
    pool_targets = {
        "generated_multilingual_examples.jsonl": 0.25,
    }
    default_target = 0.70

    merged: list[TrainingExample] = []
    sampling_weights: list[float] = []
    seen: set[tuple[str, str]] = set()
    fingerprints: list[str] = []
    policy: dict[str, dict[str, int | float]] = {}

    for path in paths:
        if not path.exists():
            continue
        fingerprints.append(_file_fingerprint(path))
        pool: list[TrainingExample] = []
        for example in load_examples(path):
            key = (example.instruction.casefold(), example.response.casefold())
            if key in seen:
                continue
            seen.add(key)
            pool.append(example)

        if not pool:
            continue

        if "generated_multilingual_examples.jsonl" in path.name:
            target = 0.25
        elif "phone_contacts_examples.jsonl" in path.name:
            target = 0.05
        else:
            target = default_target
        per_example_weight = target / len(pool)
        start = len(merged)
        merged.extend(pool)
        sampling_weights.extend([per_example_weight] * len(pool))
        policy[path.name] = {
            "examples": len(pool),
            "target_probability": target,
            "start_index": start,
        }

    total_weight = sum(sampling_weights)
    if total_weight <= 0:
        raise ValueError("instruction dataset is empty")
    sampling_weights = [weight / total_weight for weight in sampling_weights]
    return merged, fingerprints, sampling_weights, policy


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


def _instruction_batchify(
    examples: list[TrainingExample],
    tokenizer: BPETokenizer,
    block_size: int,
    batch_size: int,
    device: str,
    generator: torch.Generator,
    sampling_weights: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build supervised instruction batches with loss only on response tokens."""

    if not examples:
        raise ValueError("instruction examples cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if block_size < 8:
        raise ValueError("block_size must be large enough for instruction formatting")

    bos_id = tokenizer.stoi["<bos>"]
    eos_id = tokenizer.stoi["<eos>"]
    pad_id = tokenizer.stoi["<pad>"]

    batch_inputs: list[list[int]] = []
    batch_targets: list[list[int]] = []
    max_length = 0

    if sampling_weights is None:
        sample_indices = torch.randint(
            0,
            len(examples),
            (batch_size,),
            generator=generator,
        )
    else:
        if len(sampling_weights) != len(examples):
            raise ValueError("sampling_weights must match the instruction example count")
        weights = torch.tensor(sampling_weights, dtype=torch.float32)
        if torch.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("instruction sampling weights must be non-negative and non-zero")
        sample_indices = torch.multinomial(
            weights,
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
    for index in sample_indices.tolist():
        example = examples[index]
        prefix = (
            "<instruction>\n"
            f"{example.instruction.strip()}\n"
            "</instruction>\n"
            "<response>\n"
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        response_ids = tokenizer.encode(example.response, add_special_tokens=False)

        # Leave room for BOS and EOS while keeping the input length <= block_size.
        max_response_tokens = block_size - len(prefix_ids) - 1
        if max_response_tokens < 1:
            raise ValueError("instruction example exceeds the configured block size")
        if len(response_ids) >= max_response_tokens:
            response_ids = response_ids[: max_response_tokens - 1]

        sequence = [bos_id] + prefix_ids + response_ids + [eos_id]
        inputs = sequence[:-1]
        targets = sequence[1:]

        # Only response tokens (plus EOS) should contribute to the SFT loss.
        targets[: len(prefix_ids)] = [-100] * len(prefix_ids)

        batch_inputs.append(inputs)
        batch_targets.append(targets)
        max_length = max(max_length, len(inputs))

    input_batch = torch.full(
        (batch_size, max_length),
        pad_id,
        dtype=torch.long,
    )
    target_batch = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )
    for row, (inputs, targets) in enumerate(zip(batch_inputs, batch_targets)):
        length = len(inputs)
        input_batch[row, :length] = torch.tensor(inputs, dtype=torch.long)
        target_batch[row, :length] = torch.tensor(targets, dtype=torch.long)

    return input_batch.to(device), target_batch.to(device)


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
    batch_size: int = 8,
    checkpoint_interval: int = 500,
    learning_rate: float = 3e-4,
    instruction_path: Path | None = None,
    multilingual_instruction_path: Path | None = None,
    capability_instruction_path: Path | None = None,
    instruction_mix_ratio: float = 0.9,
) -> float:
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be greater than zero")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if not 0.0 <= instruction_mix_ratio <= 1.0:
        raise ValueError("instruction_mix_ratio must be between 0 and 1")

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_text = corpus_path.read_text(encoding="utf-8")
    instruction_paths = [
        path
        for path in (instruction_path, multilingual_instruction_path, capability_instruction_path)
        if path is not None
    ]
    instruction_enabled = any(path.exists() for path in instruction_paths)
    instruction_fingerprints: list[str] = []
    instruction_example_count = 0
    instruction_examples: list[TrainingExample] = []
    instruction_sampling_weights: list[float] | None = None
    instruction_sampling_policy: dict[str, dict[str, int | float]] = {}
    if instruction_enabled:
        (
            instruction_examples,
            instruction_fingerprints,
            instruction_sampling_weights,
            instruction_sampling_policy,
        ) = _merge_instruction_sets_weighted(instruction_paths)
        output_dir.mkdir(parents=True, exist_ok=True)
        instruction_corpus = output_dir / "instruction_corpus.txt"
        write_corpus(instruction_examples, instruction_corpus)
        instruction_example_count = len(instruction_examples)
    if len(train_text) < 32:
        raise ValueError("training corpus is too small; add more text")

    tokenizer_text = train_text
    if instruction_examples:
        tokenizer_text = (
            train_text.rstrip()
            + "\n\n"
            + (output_dir / "instruction_corpus.txt").read_text(encoding="utf-8")
        )
    tokenizer = BPETokenizer.train(
        tokenizer_text,
        vocab_size=DEFAULT_VOCAB_SIZE,
        min_frequency=DEFAULT_MIN_FREQUENCY,
    )
    train_encoded = torch.tensor(
        tokenizer.encode(train_text, add_special_tokens=True),
        dtype=torch.long,
    )
    instruction_token_count = 0
    if instruction_examples:
        instruction_text = (output_dir / "instruction_corpus.txt").read_text(encoding="utf-8")
        instruction_token_count = len(tokenizer.encode(instruction_text, add_special_tokens=False))
        if instruction_token_count <= 3:
            raise ValueError(
                "instruction corpus is too small after tokenization; add more instruction examples"
            )
    if len(train_encoded) < 4:
        raise ValueError("training corpus is too small after tokenization")

    block_size = min(DEFAULT_MODEL_CONFIG["block_size"], max(2, len(train_encoded) // 2))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    instruction_generator = torch.Generator()
    instruction_generator.manual_seed(seed + 1)
    history: list[dict[str, float | int | None]] = []
    last_loss = float("inf")
    best_validation_loss = float("inf")
    best_validation_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None

    model.train()
    for step in range(1, steps + 1):
        use_instruction_batch = (
            bool(instruction_examples)
            and torch.rand((), generator=train_generator).item() < instruction_mix_ratio
        )
        if use_instruction_batch:
            x, y = _instruction_batchify(
                instruction_examples,
                tokenizer,
                block_size,
                batch_size,
                device,
                instruction_generator,
                instruction_sampling_weights,
            )
        else:
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
                "model_version": "indoone-gpt-v2",
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
                "instruction_mix_ratio": instruction_mix_ratio,
                "instruction_token_count": instruction_token_count,
                "instruction_fingerprints": instruction_fingerprints,
                "instruction_sampling_policy": instruction_sampling_policy,
                "source_fingerprint": _file_fingerprint(corpus_path),
                "validation_fingerprint": validation_fingerprint,
                "training_text_characters": len(train_text),
                "model_config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return last_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Indoone local language model")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/train.txt"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.txt"))
    parser.add_argument("--instructions", type=Path, default=Path("data/raw/indoone_instructions.jsonl"))
    parser.add_argument("--multilingual-instructions", type=Path, default=Path("data/raw/indoone_multilingual_examples.jsonl"))
    parser.add_argument("--capability-instructions", type=Path, default=Path("data/raw/indoone_phone_contacts_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("models/indoone-small"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--instruction-mix-ratio", type=float, default=0.9)
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
        args.capability_instructions,
        args.instruction_mix_ratio,
    )
    print(f"training complete; final loss={loss:.4f}")


if __name__ == "__main__":
    main()
