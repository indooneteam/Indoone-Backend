from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.train import (
    CAPABILITY_INSTRUCTION_WEIGHT,
    CURATED_INSTRUCTION_WEIGHT,
    GENERATED_INSTRUCTION_WEIGHT,
    _instruction_batchify,
    _merge_instruction_sets_weighted,
    _prepare_instruction_examples,
    evaluate_instruction_loss,
)
from app.ai.training_data import load_examples


DEFAULT_MODEL_DIR = Path("models/indoone-small")
DEFAULT_TRAIN_INSTRUCTIONS = Path("data/processed/curated_instructions.jsonl")
DEFAULT_GENERATED_INSTRUCTIONS = Path("data/processed/generated_multilingual_examples.jsonl")
DEFAULT_VALIDATION_INSTRUCTIONS = Path("data/processed/instructions_validation.jsonl")
DEFAULT_CAPABILITY_INSTRUCTIONS = Path("data/raw/indoone_phone_contacts_examples.jsonl")
DEFAULT_STEPS = 8000
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_SEED = 4242
DEFAULT_WEIGHT_DECAY = 0.01


def run_sft(
    model_dir: Path,
    train_instructions: Path,
    generated_instructions: Path,
    validation_instructions: Path,
    capability_instructions: Path,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_interval: int,
    seed: int,
) -> dict[str, object]:
    model_path = model_dir / "indoone-small.pt"
    tokenizer_path = model_dir / "tokenizer.json"
    resume_path = model_dir / "sft_checkpoint.pt"
    base_backup = model_dir / "base_model_before_sft.pt"

    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
    if not train_instructions.is_file():
        raise FileNotFoundError(f"training instructions not found: {train_instructions}")
    if not validation_instructions.is_file():
        raise FileNotFoundError(f"validation instructions not found: {validation_instructions}")
    if steps <= 0 or batch_size <= 0 or learning_rate <= 0 or eval_interval <= 0:
        raise ValueError("steps, batch_size, learning_rate, and eval_interval must be positive")

    model_dir.mkdir(parents=True, exist_ok=True)
    if not base_backup.exists():
        shutil.copy2(model_path, base_backup)

    source_path = resume_path if resume_path.exists() else base_backup
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)

    if resume_path.exists():
        model_config = dict(checkpoint["model_config"])
        model_config.pop("model_version", None)
        model_state = checkpoint["model_state"]
        completed = int(checkpoint["step"])
        optimizer_state = checkpoint.get("optimizer_state")
        best_loss = float(checkpoint.get("best_validation_loss", float("inf")))
        best_state = checkpoint.get("best_model_state")
    else:
        model_config = dict(checkpoint["config"])
        model_config.pop("model_version", None)
        model_state = checkpoint["model_state"]
        completed = 0
        optimizer_state = None
        best_loss = float("inf")
        best_state = None

    tokenizer = BPETokenizer.load(tokenizer_path)
    curated_examples = load_examples(train_instructions)
    generated_examples = (
        load_examples(generated_instructions)
        if generated_instructions.is_file()
        else []
    )
    capability_examples = (
        load_examples(capability_instructions)
        if capability_instructions.is_file()
        else []
    )
    source_examples = [curated_examples, generated_examples, capability_examples]
    source_weights = [
        CURATED_INSTRUCTION_WEIGHT,
        GENERATED_INSTRUCTION_WEIGHT,
        CAPABILITY_INSTRUCTION_WEIGHT,
    ]
    train_examples, _fingerprints, sampling_weights, sampling_policy = _merge_instruction_sets_weighted(
        [train_instructions, generated_instructions, capability_instructions]
    )
    if not train_examples:
        raise ValueError("SFT training pool is empty")
    validation_examples = load_examples(validation_instructions)
    if not train_examples or not validation_examples:
        raise ValueError("SFT train/validation sets must both be non-empty")

    block_size = int(model_config["block_size"])
    prepared = _prepare_instruction_examples(train_examples, tokenizer, block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"SFT GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("SFT GPU unavailable; running on CPU", flush=True)

    model = IndooneTransformer(
        vocab_size=tokenizer.vocab_size,
        **model_config,
    ).to(device)
    model.load_state_dict(model_state)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if isinstance(best_state, dict):
        best_state = {key: value.clone() for key, value in best_state.items()}

    start = completed + 1
    if start > steps:
        return {"completed_step": completed, "best_validation_loss": best_loss}

    generator = torch.Generator()
    generator.manual_seed(seed + completed)

    started = time.monotonic()
    model.train()

    for step in range(start, steps + 1):
        x, y = _instruction_batchify(
            train_examples,
            tokenizer,
            block_size,
            batch_size,
            device,
            generator,
            sampling_weights=sampling_weights,
            prepared_examples=prepared,
        )
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss = float(loss.detach().cpu())

        if step == start or step % 50 == 0:
            elapsed = time.monotonic() - started
            speed = (step - start + 1) / elapsed if elapsed > 0 else 0.0
            print(
                f"SFT step {step}/{steps} | loss={train_loss:.4f} | "
                f"speed={speed:.2f} steps/s",
                flush=True,
            )

        if step % eval_interval == 0 or step == steps:
            validation_loss = evaluate_instruction_loss(
                model,
                validation_examples,
                tokenizer,
                block_size,
                batch_size,
                device,
            )
            print(f"SFT validation @ {step}: {validation_loss}", flush=True)

            if validation_loss is not None and validation_loss < best_loss:
                best_loss = float(validation_loss)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            torch.save(
                {
                    "step": step,
                    "model_config": model.config(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_state": model.state_dict(),
                    "best_validation_loss": best_loss,
                    "best_model_state": best_state,
                    "seed": seed,
                },
                resume_path,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "config": {**model_config, "model_version": "indoone-gpt-v2"},
            "model_state": model.cpu().state_dict(),
        },
        model_path,
    )

    metadata_path = model_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["post_sft"] = {
        "enabled": True,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "validation_examples": len(validation_examples),
        "training_examples": len(train_examples),
        "training_sampling_policy": sampling_policy,
        "best_validation_loss": best_loss if best_state is not None else None,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    resume_path.unlink(missing_ok=True)
    return {
        "completed_step": steps,
        "best_validation_loss": best_loss if best_state is not None else None,
        "model_path": str(model_path),
        "base_backup": str(base_backup),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated instruction SFT after base training.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-instructions", type=Path, default=DEFAULT_TRAIN_INSTRUCTIONS)
    parser.add_argument("--generated-instructions", type=Path, default=DEFAULT_GENERATED_INSTRUCTIONS)
    parser.add_argument("--validation-instructions", type=Path, default=DEFAULT_VALIDATION_INSTRUCTIONS)
    parser.add_argument("--capability-instructions", type=Path, default=DEFAULT_CAPABILITY_INSTRUCTIONS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    print(
        json.dumps(
            run_sft(
                model_dir=args.model_dir,
                train_instructions=args.train_instructions,
                generated_instructions=args.generated_instructions,
                validation_instructions=args.validation_instructions,
                capability_instructions=args.capability_instructions,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                eval_interval=args.eval_interval,
                seed=args.seed,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
