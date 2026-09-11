from pathlib import Path

import pytest

import scripts.promote_model as promotion


def _prepare_candidate(
    tmp_path: Path,
    report: str = '{"evaluation":{"model_version":"indoone-gpt-v1","loss":1.0,"perplexity":2.0}}\n',
) -> tuple[Path, Path, Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "indoone-small.pt").write_bytes(b"checkpoint")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "evaluation_report.json").write_text(report, encoding="utf-8")
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry.json"
    return model_dir, cases, registry, model_dir / "evaluation_report.json"


def test_promotion_requires_behavioral_gate(monkeypatch, tmp_path: Path) -> None:
    model_dir, cases, registry, _ = _prepare_candidate(tmp_path)

    monkeypatch.setattr(
        promotion,
        "run_behavioral_eval",
        lambda **_: {"gate": {"overall_pass": False}},
    )

    with pytest.raises(ValueError, match="behavioral gate"):
        promotion.promote_model(
            version="v1",
            model_dir=model_dir,
            behavior_cases=cases,
            registry_path=registry,
        )


def test_successful_candidate_is_promoted(monkeypatch, tmp_path: Path) -> None:
    model_dir, cases, registry, _ = _prepare_candidate(tmp_path)

    monkeypatch.setattr(
        promotion,
        "run_behavioral_eval",
        lambda **_: {"gate": {"overall_pass": True}},
    )

    promoted = promotion.promote_model(
        version="v1",
        model_dir=model_dir,
        behavior_cases=cases,
        registry_path=registry,
    )

    assert promoted.version == "v1"
    assert promoted.status == "active"
    assert promoted.parent_version is None
    report = (model_dir / "promotion_report.json").read_text(encoding="utf-8")
    assert '"promoted": true' in report
    assert '"benchmark_version": "v1"' in report


def test_promotion_rejects_missing_artifacts(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "evaluation_report.json").write_text(
        '{"evaluation":{"model_version":"indoone-gpt-v1","loss":1.0,"perplexity":2.0}}\n',
        encoding="utf-8",
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="promotion artifacts are missing"):
        promotion.promote_model(
            version="v1",
            model_dir=model_dir,
            behavior_cases=cases,
            registry_path=tmp_path / "registry.json",
        )


def test_promotion_rejects_missing_model_version(tmp_path: Path) -> None:
    model_dir, cases, registry, _ = _prepare_candidate(
        tmp_path,
        '{"evaluation":{"loss":1.0,"perplexity":2.0}}\n',
    )

    with pytest.raises(ValueError, match="missing model_version"):
        promotion.promote_model(
            version="v1",
            model_dir=model_dir,
            behavior_cases=cases,
            registry_path=registry,
        )
