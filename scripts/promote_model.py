from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from app.ai.behavior_eval import run_behavioral_eval
from app.ai.model_registry import ModelRecord, promote_candidate


def _load_evaluation(report_path: Path) -> tuple[dict[str, object], float, float]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load evaluation report: {report_path}") from exc
    if not isinstance(report, dict):
        raise ValueError("evaluation report must contain a JSON object")

    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation report is missing the evaluation object")

    model_version = evaluation.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("evaluation report is missing model_version")

    try:
        loss = float(evaluation["loss"])
        perplexity = float(evaluation["perplexity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluation report is missing valid loss/perplexity metrics") from exc

    if not math.isfinite(loss) or not math.isfinite(perplexity):
        raise ValueError("evaluation metrics must be finite")
    if loss < 0 or perplexity < 0:
        raise ValueError("evaluation metrics cannot be negative")
    return evaluation, loss, perplexity


def _require_artifacts(model_dir: Path, behavior_cases: Path) -> None:
    required = (
        model_dir / "indoone-small.pt",
        model_dir / "tokenizer.json",
        behavior_cases,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("promotion artifacts are missing: " + ", ".join(missing))


def promote_model(
    version: str,
    model_dir: Path,
    behavior_cases: Path,
    registry_path: Path,
    evaluation_report_path: Path | None = None,
) -> ModelRecord:
    _require_artifacts(model_dir, behavior_cases)
    report_path = evaluation_report_path or (model_dir / "evaluation_report.json")
    evaluation, loss, perplexity = _load_evaluation(report_path)

    behavioral = run_behavioral_eval(
        cases_path=behavior_cases,
        checkpoint_path=model_dir / "indoone-small.pt",
        tokenizer_path=model_dir / "tokenizer.json",
        temperature=0.0,
    )
    gate = behavioral.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("behavioral evaluation did not return a gate summary")
    gate_passed = gate.get("overall_pass")
    if not isinstance(gate_passed, bool):
        raise ValueError("behavioral evaluation gate must contain a boolean overall_pass")

    candidate = ModelRecord(
        version=version.strip(),
        model_dir=str(model_dir),
        checkpoint=str(model_dir / "indoone-small.pt"),
        tokenizer=str(model_dir / "tokenizer.json"),
        loss=loss,
        perplexity=perplexity,
        status="candidate",
        behavioral_gate_passed=gate_passed,
        benchmark_version="v1",
    )
    promoted = promote_candidate(registry_path, candidate)

    output = model_dir / "promotion_report.json"
    output.write_text(
        json.dumps(
            {
                "version": version,
                "evaluation": evaluation,
                "behavioral_gate": gate,
                "promotion": {
                    "benchmark_version": promoted.benchmark_version,
                    "parent_version": promoted.parent_version,
                    "promoted": promoted.status == "active",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return promoted


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and promote an Indoone model candidate")
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models/indoone-small"))
    parser.add_argument("--behavior-cases", type=Path, default=Path("data/eval/behavior.jsonl"))
    parser.add_argument("--registry", type=Path, default=Path("models/registry.json"))
    parser.add_argument("--evaluation-report", type=Path)
    args = parser.parse_args()

    promoted = promote_model(
        version=args.version,
        model_dir=args.model_dir,
        behavior_cases=args.behavior_cases,
        registry_path=args.registry,
        evaluation_report_path=args.evaluation_report,
    )
    print(f"promoted model {promoted.version} as {promoted.status}")


if __name__ == "__main__":
    main()
