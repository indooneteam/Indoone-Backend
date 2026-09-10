from __future__ import annotations

from app.ai.benchmark import build_comparison_report, compare_benchmark_reports
from app.ai.model_registry import ModelRecord, should_promote


def _report(*, version: str = "v1", loss: float = 1.0, perplexity: float = 2.0, overall_pass: bool = True) -> dict[str, object]:
    return {
        "benchmark_version": version,
        "model": {"checkpoint": "model.pt", "tokenizer": "tokenizer.json"},
        "language_metrics": {"loss": loss, "perplexity": perplexity},
        "behavioral_gate": {"overall_pass": overall_pass},
        "overall_pass": overall_pass,
    }


def _record(*, version: str, loss: float, perplexity: float, status: str = "candidate", benchmark_version: str = "v1", behavioral: bool = True, parent_version: str | None = None) -> ModelRecord:
    return ModelRecord(
        version=version,
        model_dir="models/indoone-small",
        checkpoint="models/indoone-small/indoone-small.pt",
        tokenizer="models/indoone-small/tokenizer.json",
        loss=loss,
        perplexity=perplexity,
        status=status,
        behavioral_gate_passed=behavioral,
        parent_version=parent_version,
        benchmark_version=benchmark_version,
    )


def test_v1_benchmark_schema_is_frozen() -> None:
    report = _report()

    assert set(report) == {
        "benchmark_version",
        "model",
        "language_metrics",
        "behavioral_gate",
        "overall_pass",
    }
    assert set(report["language_metrics"]) == {"loss", "perplexity"}
    assert report["benchmark_version"] == "v1"


def test_v1_comparison_requires_strict_improvement_and_behavioral_pass() -> None:
    baseline = _report(loss=1.0, perplexity=2.0, overall_pass=True)
    candidate = _report(loss=0.9, perplexity=1.9, overall_pass=True)

    result = compare_benchmark_reports(baseline, candidate)

    assert result["benchmark_version"] == "v1"
    assert result["overall_improved"] is True


def test_v1_rejects_benchmark_version_changes() -> None:
    baseline = _report(version="v1")
    candidate = _report(version="v2")

    try:
        compare_benchmark_reports(baseline, candidate)
    except ValueError as exc:
        assert str(exc) == "benchmark versions must match"
    else:
        raise AssertionError("benchmark version changes must not silently pass")


def test_v1_promotion_policy_requires_both_metrics_to_improve() -> None:
    current = _record(version="v1", loss=1.0, perplexity=2.0, status="active")
    better_loss_only = _record(version="v2", loss=0.9, perplexity=2.0)
    better_both = _record(version="v3", loss=0.9, perplexity=1.9)

    assert should_promote(better_loss_only, current) is False
    assert should_promote(better_both, current) is True


def test_build_comparison_report_preserves_frozen_v1_decision_shape(tmp_path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(__import__("json").dumps(_report(loss=1.0, perplexity=2.0)) + "\n", encoding="utf-8")
    candidate_path.write_text(__import__("json").dumps(_report(loss=0.9, perplexity=1.9)) + "\n", encoding="utf-8")

    result = build_comparison_report(baseline_path, candidate_path)

    assert set(result) == {
        "benchmark_version",
        "baseline_pass",
        "candidate_pass",
        "behavioral_regression_free",
        "loss_delta",
        "perplexity_delta",
        "improved",
        "overall_improved",
        "baseline_behavioral_gate",
        "candidate_behavioral_gate",
    }
