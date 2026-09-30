from pathlib import Path

import pytest

from app.ai.training.model_registry import (
    ModelRecord,
    active_record,
    load_records,
    promote_candidate,
    should_promote,
)


def record(
    version: str,
    loss: float,
    perplexity: float,
    status: str = "candidate",
    behavioral_gate_passed: bool = True,
    benchmark_version: str = "v1",
    revision: str | None = None,
    max_metric_regression: float = 0.0,
) -> ModelRecord:
    return ModelRecord(
        version=version,
        model_dir=f"models/{version}",
        checkpoint=f"models/{version}/indoone-small.pt",
        tokenizer=f"models/{version}/tokenizer.json",
        loss=loss,
        perplexity=perplexity,
        status=status,
        behavioral_gate_passed=behavioral_gate_passed,
        benchmark_version=benchmark_version,
        revision=revision or f"rev-{version}",
        max_metric_regression=max_metric_regression,
    )


def test_first_candidate_can_be_promoted(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    candidate = record("v1", 1.5, 4.5, revision="initial")

    assert should_promote(candidate, None)
    promoted = promote_candidate(registry, candidate)

    assert promoted.status == "active"
    assert promoted.parent_version is None
    assert promoted.benchmark_version == "v1"
    assert promoted.revision == "initial"
    assert active_record(registry) == promoted


def test_candidate_must_improve_both_metrics(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    promote_candidate(registry, record("v1", 1.5, 4.5, revision="rev-1"))

    better = record("v1", 1.4, 4.4, revision="rev-2")
    worse_loss = record("v1", 1.6, 4.0, revision="rev-3")
    worse_perplexity = record("v1", 1.4, 4.6, revision="rev-4")

    assert should_promote(better, active_record(registry))
    assert not should_promote(worse_loss, active_record(registry))
    assert not should_promote(worse_perplexity, active_record(registry))


def test_behavioral_gate_is_required_for_promotion(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    candidate = record("v1", 1.0, 2.0, behavioral_gate_passed=False)

    assert should_promote(candidate, None) is False
    with pytest.raises(ValueError, match="behavioral gate"):
        promote_candidate(registry, candidate)


def test_same_version_update_retires_previous_revision(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    promote_candidate(registry, record("v1", 1.5, 4.5, revision="rev-1"))
    promoted = promote_candidate(registry, record("v1", 1.4, 4.4, revision="rev-2"))

    records = {(item.version, item.revision): item for item in load_records(registry)}
    assert records[("v1", "rev-1")].status == "retired"
    assert records[("v1", "rev-2")].status == "active"
    assert promoted.parent_version == "v1"
    assert active_record(registry).revision == "rev-2"
    assert active_record(registry).version == "v1"



def test_capability_candidate_can_tolerate_small_metric_regression() -> None:
    current = record("v1", 1.0, 4.0, status="active", revision="rev-1")
    candidate = record(
        "v1",
        1.01,
        4.04,
        revision="rev-2",
        max_metric_regression=0.02,
    )

    assert should_promote(candidate, current)


def test_capability_candidate_cannot_exceed_declared_metric_tolerance() -> None:
    current = record("v1", 1.0, 4.0, status="active", revision="rev-1")
    candidate = record(
        "v1",
        1.03,
        4.04,
        revision="rev-2",
        max_metric_regression=0.02,
    )

    assert not should_promote(candidate, current)

def test_different_version_update_is_still_supported(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    promote_candidate(registry, record("v1", 1.5, 4.5, revision="rev-1"))
    promoted = promote_candidate(registry, record("v2", 1.4, 4.4, revision="rev-2"))

    assert promoted.status == "active"
    assert promoted.parent_version == "v1"


def test_non_candidate_and_negative_metrics_are_rejected(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"

    with pytest.raises(ValueError, match="only candidate"):
        promote_candidate(registry, record("v1", 1.0, 2.0, status="active"))

    with pytest.raises(ValueError, match="cannot be negative"):
        should_promote(record("v2", -1.0, 2.0), None)


def test_invalid_benchmark_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported benchmark version"):
        should_promote(record("v1", 1.0, 2.0, benchmark_version="v2"), None)


def test_benchmark_version_must_match_active_model() -> None:
    current = record("v1", 1.5, 4.5, status="active", benchmark_version="v1", revision="rev-1")
    candidate = record("v1", 1.4, 4.4, benchmark_version="v2", revision="rev-2")

    with pytest.raises(ValueError, match="benchmark versions"):
        should_promote(candidate, current)


def test_same_version_same_revision_is_rejected() -> None:
    current = record("v1", 1.5, 4.5, status="active", revision="rev-1")
    candidate = record("v1", 1.4, 4.4, revision="rev-1")

    with pytest.raises(ValueError, match="candidate revision"):
        should_promote(candidate, current)
