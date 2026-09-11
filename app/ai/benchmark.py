from __future__ import annotations

import argparse
import json
import math
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


def load_benchmark_report(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load benchmark report: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("benchmark report must contain a JSON object")
    return payload


def _validate_report(report: dict[str, object]) -> tuple[str, dict[str, object], dict[str, object], bool]:
    try:
        version = report["benchmark_version"]
        metrics = report["language_metrics"]
        gate = report["behavioral_gate"]
        overall_pass = report["overall_pass"]
        if not isinstance(version, str) or not version.strip():
            raise TypeError
        if not isinstance(metrics, dict) or not isinstance(gate, dict):
            raise TypeError
        if not isinstance(overall_pass, bool):
            raise TypeError
        loss = float(metrics["loss"])
        perplexity = float(metrics["perplexity"])
        if not math.isfinite(loss) or not math.isfinite(perplexity):
            raise ValueError
        if loss < 0 or perplexity < 0:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid benchmark report shape") from exc
    return version, metrics, gate, overall_pass


def compare_benchmark_reports(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Compare two benchmark reports for a safe model-improvement decision."""
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("invalid benchmark report shape")

    baseline_version, baseline_metrics, baseline_gate, baseline_pass = _validate_report(baseline)
    candidate_version, candidate_metrics, candidate_gate, candidate_pass = _validate_report(candidate)

    if baseline_version != candidate_version:
        raise ValueError("benchmark versions must match")
    if baseline_version != "v1":
        raise ValueError("unsupported benchmark version")

    try:
        baseline_loss = float(baseline_metrics["loss"])
        candidate_loss = float(candidate_metrics["loss"])
        baseline_perplexity = float(baseline_metrics["perplexity"])
        candidate_perplexity = float(candidate_metrics["perplexity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid benchmark report shape") from exc

    loss_delta = candidate_loss - baseline_loss
    perplexity_delta = candidate_perplexity - baseline_perplexity
    improved = candidate_loss < baseline_loss and candidate_perplexity < baseline_perplexity
    return {
        "benchmark_version": "v1",
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


def build_comparison_report(baseline_path: Path, candidate_path: Path) -> dict[str, object]:
    """Load two saved benchmark reports and produce the promotion decision."""
    return compare_benchmark_reports(
        load_benchmark_report(baseline_path),
        load_benchmark_report(candidate_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repeatable Indoone core AI benchmark")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--candidate-report", type=Path)
    args = parser.parse_args()

    if (args.baseline_report is None) != (args.candidate_report is None):
        parser.error("--baseline-report and --candidate-report must be provided together")

    if args.baseline_report is not None and args.candidate_report is not None:
        report = build_comparison_report(args.baseline_report, args.candidate_report)
    else:
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
