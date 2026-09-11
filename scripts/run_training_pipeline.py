from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare, train, publish, and evaluate an Indoone model in one command."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    if shutil.which("nvidia-smi") is None:
        raise SystemExit(
            "NVIDIA GPU not detected. Run this pipeline on a CUDA-capable machine; "
            "the local CPU environment is intentionally not used for the full training run."
        )

    python = sys.executable
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
            "app.ai.train",
            "--corpus",
            "data/processed/train.txt",
            "--validation",
            "data/processed/validation.txt",
            "--instructions",
            "data/raw/indoone_generated_instructions.jsonl",
            "--multilingual-instructions",
            "data/raw/indoone_multilingual_examples.jsonl",
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
            "--seed",
            str(args.seed),
        ]
    )

    if not args.skip_upload:
        run([python, "scripts/upload_model_to_b2.py"])

    if not args.skip_eval:
        run(
            [
                python,
                "scripts/evaluate_model.py",
                "--model",
                "models/indoone-small/indoone-small.pt",
                "--tokenizer",
                "models/indoone-small/tokenizer.json",
            ]
        )


if __name__ == "__main__":
    main()
