from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ai.behavior_eval import run_behavioral_eval
from app.ai.model_registry import ModelRecord, promote_candidate


def promote_model(
    version: str,
    model_dir: Path,
    behavior_cases: Path,
    registry_path: Path,
    evaluation_report_path: Path | None = None,
) -> ModelRecord:
    report_path = evaluation_report_path or (model_dir / "evaluation_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation report is missing the evaluation object")

    loss = float(evaluation["loss"])
    perplexity = float(evaluation["perplexity"])
    behavioral = run_behavioral_eval(
        cases_path=behavior_cases,
        checkpoint_path=model_dir / "indoone-small.pt",
        tokenizer_path=model_dir / "tokenizer.json",
        temperature=0.0,
    )
    gate = behavioral["gate"]
    if not isinstance(gate, dict):
        raise ValueError("behavioral evaluation did not return a gate summary")

    candidate = ModelRecord(
        version=version,
        model_dir=str(model_dir),
        checkpoint=str(model_dir / "indoone-small.pt"),
        tokenizer=str(model_dir / "tokenizer.json"),
        loss=loss,
        perplexity=perplexity,
        status="candidate",
        behavioral_gate_passed=bool(gate["overall_pass"]),
    )
    promoted = promote_candidate(registry_path, candidate)

    output = model_dir / "promotion_report.json"
    output.write_text(
        json.dumps(
            {
                "version": version,
                "evaluation": evaluation,
                "behavioral_gate": gate,
                "promoted": promoted.status == "active",
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
