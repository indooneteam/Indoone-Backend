from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from app.ai.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_INSTRUCTION_MIX_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_TRAINING_STEPS,
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, train, evaluate, and optionally publish an Indoone model."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--instruction-mix-ratio", type=float, default=DEFAULT_INSTRUCTION_MIX_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not 0.0 <= args.instruction_mix_ratio <= 1.0:
        raise SystemExit("--instruction-mix-ratio must be between 0 and 1")

    python = sys.executable
    eval_prompts = Path("data/evaluation/behavior_prompts.jsonl")
    curated_instructions = Path("data/processed/curated_instructions.jsonl")
    generated_instructions = Path("data/processed/generated_multilingual_examples.jsonl")
    validation_instructions = Path("data/processed/instructions_validation.jsonl")
    allow_cpu_training = os.getenv("INDOONE_ALLOW_CPU_TRAINING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Build deterministic augmentation into generated/processed artifacts only.
    run([python, "scripts/build_multilingual_training_pack.py"])

    # Readiness audit includes the generated augmentation but never rewrites raw sources.
    run(
        [
            python,
            "scripts/audit_dataset.py",
            "--corpus",
            "data/processed/generated_multilingual_corpus.txt",
            "--instructions",
            "data/raw/indoone_instructions.jsonl",
            "--instructions",
            "data/raw/core_instruction_seed.jsonl",
            "--instructions",
            "data/raw/indoone_multilingual_examples.jsonl",
            "--instructions",
            "data/raw/indoone_phone_contacts_examples.jsonl",
            "--instructions",
            str(generated_instructions),
            "--multilingual",
            str(generated_instructions),
        ]
    )

    if shutil.which("nvidia-smi") is None:
        if not allow_cpu_training:
            raise SystemExit(
                "NVIDIA GPU not detected. Run this pipeline on a CUDA-capable machine, "
                "or explicitly set INDOONE_ALLOW_CPU_TRAINING=true for a hosted CPU training run."
            )
        print(
            "NVIDIA GPU not detected; explicit CPU training mode is enabled for this run.",
            flush=True,
        )

    run(
        [
            python,
            "-m",
            "scripts.prepare_dataset",
            "--source",
            "data/raw/indoone_corpus.txt",
            "--output-dir",
            "data/processed",
        ]
    )

    # Curated instruction data is kept in its own pool so it can be deliberately
    # oversampled during supervised training.
    run(
        [
            python,
            "scripts/assemble_training_dataset.py",
            "--source",
            "data/raw/indoone_instructions.jsonl",
            "--source",
            "data/raw/core_instruction_seed.jsonl",
            "--source",
            "data/raw/indoone_multilingual_examples.jsonl",
            "--output",
            str(curated_instructions),
            "--validation-output",
            str(validation_instructions),
            "--validation-ratio",
            "0.2",
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "scripts/validate_dataset_quality.py",
            "--source",
            str(curated_instructions),
            "--source",
            str(validation_instructions),
            "--source",
            str(generated_instructions),
            "--source",
            "data/raw/indoone_phone_contacts_examples.jsonl",
        ]
    )
    run([python, "scripts/validate_training_manifest.py"])
    run([python, "scripts/validate_final_training_recipe.py"])

    run(
        [
            python,
            "-m",
            "app.ai.train",
            "--corpus",
            "data/processed/train.txt",
            "--validation",
            "data/processed/validation.txt",
            "--instructions",
            str(curated_instructions),
            "--multilingual-instructions",
            str(generated_instructions),
            "--capability-instructions",
            "data/raw/indoone_phone_contacts_examples.jsonl",
            "--instruction-validation",
            str(validation_instructions),
            "--output",
            "models/indoone-small",
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--checkpoint-interval",
            str(args.checkpoint_interval),
            "--learning-rate",
            str(args.learning_rate),
            "--instruction-mix-ratio",
            str(args.instruction_mix_ratio),
            "--seed",
            str(args.seed),
        ]
    )

    if not args.skip_upload:
        run([python, "scripts/upload_model_to_b2.py"])

    if not args.skip_eval:
        command = [
            python,
            "scripts/evaluate_model.py",
            "--model",
            "models/indoone-small/indoone-small.pt",
            "--tokenizer",
            "models/indoone-small/tokenizer.json",
        ]
        if eval_prompts.is_file():
            command.extend(["--prompts", str(eval_prompts)])
        run(command)
        run(
            [
                python,
                "-m",
                "app.ai.evaluate",
                "--checkpoint",
                "models/indoone-small/indoone-small.pt",
                "--tokenizer",
                "models/indoone-small/tokenizer.json",
                "--corpus",
                "data/processed/test.txt",
                "--output",
                "models/indoone-small/evaluation_report.json",
            ]
        )


if __name__ == "__main__":
    main()
