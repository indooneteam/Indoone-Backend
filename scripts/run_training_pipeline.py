from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


RAW_TRAINING_FILES = (
    Path("data/raw/indoone_corpus.txt"),
    Path("data/raw/indoone_instructions.jsonl"),
    Path("data/raw/indoone_multilingual_examples.jsonl"),
)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def restore_curated_raw_sources() -> None:
    """Undo readiness-only synthetic augmentation before the actual training run."""
    run(["git", "checkout", "--", *(str(path) for path in RAW_TRAINING_FILES)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, validate, train, evaluate, and optionally publish an Indoone model."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--instruction-mix-ratio",
        type=float,
        default=0.9,
        help="Fraction of training steps sampled from curated instruction data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not 0.0 <= args.instruction_mix_ratio <= 1.0:
        raise SystemExit("--instruction-mix-ratio must be between 0 and 1")

    python = sys.executable
    eval_prompts = Path("data/evaluation/behavior_prompts.jsonl")
    assembled_instructions = Path("data/processed/instructions_train.jsonl")
    assembled_validation = Path("data/processed/instructions_validation.jsonl")
    allow_cpu_training = os.getenv("INDOONE_ALLOW_CPU_TRAINING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Synthetic multilingual augmentation is a readiness check, not the general-language
    # corpus for the final small-model run. Restore the curated raw sources before training.
    run([python, "scripts/build_multilingual_training_pack.py"])
    run([python, "scripts/audit_dataset.py", "--json"])
    restore_curated_raw_sources()

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
    run(
        [
            python,
            "-m",
            "scripts.assemble_training_dataset",
            "--source",
            "data/raw/indoone_instructions.jsonl",
            "--source",
            "data/raw/core_instruction_seed.jsonl",
            "--source",
            "data/raw/indoone_multilingual_examples.jsonl",
            "--source",
            "data/raw/indoone_phone_contacts_examples.jsonl",
            "--output",
            str(assembled_instructions),
            "--validation-output",
            str(assembled_validation),
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "scripts/validate_dataset_quality.py",
            "--source",
            str(assembled_instructions),
            "--source",
            str(assembled_validation),
        ]
    )
    run([python, "scripts/validate_training_manifest.py"])
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
            str(assembled_instructions),
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
