from pathlib import Path

import pytest

import scripts.promote_model as promotion


def test_promotion_requires_behavioral_gate(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    report = model_dir / "evaluation_report.json"
    report.write_text(
        '{"evaluation":{"loss":1.0,"perplexity":2.0}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        promotion,
        "run_behavioral_eval",
        lambda **_: {"gate": {"overall_pass": False}},
    )

    with pytest.raises(ValueError, match="behavioral gate"):
        promotion.promote_model(
            version="v1",
            model_dir=model_dir,
            behavior_cases=tmp_path / "cases.jsonl",
            registry_path=tmp_path / "registry.json",
        )


def test_successful_candidate_is_promoted(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "indoone-small.pt").write_bytes(b"checkpoint")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_dir / "evaluation_report.json").write_text(
        '{"evaluation":{"loss":1.0,"perplexity":2.0}}\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        promotion,
        "run_behavioral_eval",
        lambda **_: {"gate": {"overall_pass": True}},
    )

    promoted = promotion.promote_model(
        version="v1",
        model_dir=model_dir,
        behavior_cases=tmp_path / "cases.jsonl",
        registry_path=tmp_path / "registry.json",
    )

    assert promoted.version == "v1"
    assert promoted.status == "active"
    report = (model_dir / "promotion_report.json").read_text(encoding="utf-8")
    assert '"promoted": true' in report
