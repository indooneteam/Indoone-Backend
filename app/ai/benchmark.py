from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ai.behavior_eval import CATEGORIES, run_behavioral_eval
from app.ai.evaluate import evaluate_checkpoint

DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")
DEFAULT_CORPUS = Path("data/processed/test.txt")
DEFAULT_CASES = Path("data/eval/behavior.jsonl")


def build_benchmark_report(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    corpus_path: Path = DEFAULT_CORPUS,
    cases_path: Path = DEFAULT_CASES,
    batch_size: int = 16,
    max_new_tokens: int = 80,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")

    language_metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        corpus_path=corpus_path,
        batch_size=batch_size,
    )
    behavioral = run_behavioral_eval(
        cases_path=cases_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    gate = behavioral["gate"]
    assert isinstance(gate, dict)

    return {
        "benchmark_version": "v1",
        "model": {
            "checkpoint": str(checkpoint_path),
            "tokenizer": str(tokenizer_path),
        },
        "language_metrics": language_metrics,
        "behavioral_gate": gate,
        "overall_pass": bool(gate["overall_pass"]),
    }


def compare_benchmark_reports(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Compare two benchmark reports for a safe model-improvement decision."""
    try:
        baseline_metrics = baseline["language_metrics"]
        candidate_metrics = candidate["language_metrics"]
        baseline_gate = baseline["behavioral_gate"]
        candidate_gate = candidate["behavioral_gate"]
        if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
            raise TypeError
        if not isinstance(baseline_gate, dict) or not isinstance(candidate_gate, dict):
            raise TypeError
        baseline_loss = float(baseline_metrics["loss"])
        candidate_loss = float(candidate_metrics["loss"])
        baseline_perplexity = float(baseline_metrics["perplexity"])
        candidate_perplexity = float(candidate_metrics["perplexity"])
        baseline_pass = bool(baseline["overall_pass"])
        candidate_pass = bool(candidate["overall_pass"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid benchmark report shape") from exc

    loss_delta = candidate_loss - baseline_loss
    perplexity_delta = candidate_perplexity - baseline_perplexity
    improved = candidate_loss < baseline_loss and candidate_perplexity < baseline_perplexity
    return {
        "baseline_pass": baseline_pass,
        "candidate_pass": candidate_pass,
        "behavioral_regression_free": candidate_pass,
        "loss_delta": loss_delta,
        "perplexity_delta": perplexity_delta,
        "improved": improved,
        "overall_improved": candidate_pass and improved,
        "baseline_behavioral_gate": baseline_gate,
        "candidate_behavioral_gate": candidate_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repeatable Indoone core AI benchmark")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_benchmark_report(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        corpus_path=args.corpus,
        cases_path=args.cases,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
