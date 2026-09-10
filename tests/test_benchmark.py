import json
from pathlib import Path

import app.ai.benchmark as benchmark


def test_build_benchmark_report_combines_language_and_behavior_metrics(monkeypatch) -> None:
    def fake_evaluate_checkpoint(**kwargs):
        assert kwargs["batch_size"] == 8
        return {"model_version": "indoone-gpt-v1", "loss": 1.25, "perplexity": 3.49}

    def fake_run_behavioral_eval(**kwargs):
        assert kwargs["max_new_tokens"] == 24
        return {
            "cases": [{"id": "instruction", "passed": True}],
            "gate": {
                "overall_pass": True,
                "case_count": 5,
                "passed_cases": 5,
                "category_pass": {category: True for category in benchmark.CATEGORIES},
                "required_categories": list(benchmark.CATEGORIES),
            },
        }

    monkeypatch.setattr(benchmark, "evaluate_checkpoint", fake_evaluate_checkpoint)
    monkeypatch.setattr(benchmark, "run_behavioral_eval", fake_run_behavioral_eval)

    report = benchmark.build_benchmark_report(
        checkpoint_path=Path("candidate.pt"),
        tokenizer_path=Path("tokenizer.json"),
        corpus_path=Path("test.txt"),
        cases_path=Path("behavior.jsonl"),
        batch_size=8,
        max_new_tokens=24,
    )

    assert report["benchmark_version"] == "v1"
    assert report["language_metrics"]["loss"] == 1.25
    assert report["behavioral_gate"]["passed_cases"] == 5
    assert report["overall_pass"] is True


def test_build_benchmark_report_rejects_invalid_limits() -> None:
    import pytest

    with pytest.raises(ValueError, match="batch_size"):
        benchmark.build_benchmark_report(batch_size=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        benchmark.build_benchmark_report(max_new_tokens=0)


def _report(loss: float, perplexity: float, overall_pass: bool) -> dict[str, object]:
    return {
        "benchmark_version": "v1",
        "language_metrics": {"loss": loss, "perplexity": perplexity},
        "behavioral_gate": {"overall_pass": overall_pass, "category_pass": {}},
        "overall_pass": overall_pass,
    }


def test_compare_benchmark_reports_requires_behavioral_pass_and_lower_metrics() -> None:
    comparison = benchmark.compare_benchmark_reports(
        _report(2.0, 6.0, True),
        _report(1.5, 4.0, True),
    )

    assert comparison["loss_delta"] == -0.5
    assert comparison["perplexity_delta"] == -2.0
    assert comparison["improved"] is True
    assert comparison["overall_improved"] is True


def test_compare_benchmark_reports_rejects_behavioral_regression() -> None:
    comparison = benchmark.compare_benchmark_reports(
        _report(2.0, 6.0, True),
        _report(1.5, 4.0, False),
    )

    assert comparison["improved"] is True
    assert comparison["overall_improved"] is False
    assert comparison["behavioral_regression_free"] is False


def test_compare_benchmark_reports_rejects_mismatched_version() -> None:
    baseline = _report(2.0, 6.0, True)
    candidate = _report(1.5, 4.0, True)
    candidate["benchmark_version"] = "v2"

    import pytest

    with pytest.raises(ValueError, match="benchmark versions"):
        benchmark.compare_benchmark_reports(baseline, candidate)


def test_compare_benchmark_reports_requires_valid_shape() -> None:
    import pytest

    with pytest.raises(ValueError, match="invalid benchmark report shape"):
        benchmark.compare_benchmark_reports({}, _report(1.0, 2.0, True))


def test_build_comparison_report_loads_saved_reports(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(_report(2.0, 6.0, True)), encoding="utf-8")
    candidate_path.write_text(json.dumps(_report(1.5, 4.0, True)), encoding="utf-8")

    comparison = benchmark.build_comparison_report(baseline_path, candidate_path)

    assert comparison["overall_improved"] is True
    assert comparison["loss_delta"] == -0.5


def test_load_benchmark_report_rejects_invalid_json(tmp_path: Path) -> None:
    import pytest

    path = tmp_path / "broken.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="unable to load benchmark report"):
        benchmark.load_benchmark_report(path)
