from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ai.evaluate import evaluate_checkpoint
from app.ai.train import train
from scripts.prepare_dataset import prepare_dataset


def train_and_evaluate(
    source: Path,
    processed_dir: Path,
    model_dir: Path,
    steps: int,
    seed: int,
    batch_size: int,
    checkpoint_interval: int,
    learning_rate: float,
) -> dict:
    dataset_stats = prepare_dataset(
        source=source,
        output_dir=processed_dir,
        train_ratio=0.8,
        validation_ratio=0.1,
    )

    train_loss = train(
        corpus_path=processed_dir / "train.txt",
        output_dir=model_dir,
        steps=steps,
        seed=seed,
        validation_path=processed_dir / "validation.txt",
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        learning_rate=learning_rate,
    )

    metrics = evaluate_checkpoint(
        checkpoint_path=model_dir / "indoone-small.pt",
        tokenizer_path=model_dir / "tokenizer.json",
        corpus_path=processed_dir / "test.txt",
        batch_size=batch_size,
    )

    report = {
        "dataset": dataset_stats,
        "training": {
            "steps": steps,
            "seed": seed,
            "batch_size": batch_size,
            "checkpoint_interval": checkpoint_interval,
            "learning_rate": learning_rate,
            "final_train_loss": train_loss,
        },
        "evaluation": metrics,
    }
    report_path = model_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare data, train the Indoone local model, and evaluate its test split"
    )
    parser.add_argument("--source", type=Path, default=Path("data/raw/indoone_corpus.txt"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/indoone-small"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    report = train_and_evaluate(
        source=args.source,
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
