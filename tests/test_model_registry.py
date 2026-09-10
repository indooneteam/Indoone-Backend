from pathlib import Path

import pytest

from app.ai.model_registry import ModelRecord, active_record, load_records, promote_candidate, should_promote


def record(version: str, loss: float, perplexity: float, status: str = "candidate") -> ModelRecord:
    return ModelRecord(
        version=version,
        model_dir=f"models/{version}",
        checkpoint=f"models/{version}/indoone-small.pt",
        tokenizer=f"models/{version}/tokenizer.json",
        loss=loss,
        perplexity=perplexity,
        status=status,
    )


def test_first_candidate_can_be_promoted(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    candidate = record("v1", 1.5, 4.5)

    assert should_promote(candidate, None)
    promoted = promote_candidate(registry, candidate)

    assert promoted.status == "active"
    assert active_record(registry) == promoted


def test_candidate_must_improve_both_metrics(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    promote_candidate(registry, record("v1", 1.5, 4.5))

    better = record("v2", 1.4, 4.4)
    worse_loss = record("v3", 1.6, 4.0)
    worse_perplexity = record("v4", 1.4, 4.6)

    assert should_promote(better, active_record(registry))
    assert not should_promote(worse_loss, active_record(registry))
    assert not should_promote(worse_perplexity, active_record(registry))


def test_promotion_retires_previous_active_model(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    promote_candidate(registry, record("v1", 1.5, 4.5))
    promote_candidate(registry, record("v2", 1.4, 4.4))

    records = {item.version: item for item in load_records(registry)}
    assert records["v1"].status == "retired"
    assert records["v2"].status == "active"
    assert active_record(registry).version == "v2"


def test_non_candidate_and_negative_metrics_are_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"

    with pytest.raises(ValueError, match="only candidate"):
        promote_candidate(registry, record("v1", 1.0, 2.0, status="active"))

    with pytest.raises(ValueError, match="cannot be negative"):
        should_promote(record("v2", -1.0, 2.0), None)
