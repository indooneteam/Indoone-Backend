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
