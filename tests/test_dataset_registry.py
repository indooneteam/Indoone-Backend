from pathlib import Path

import pytest

from app.ai.dataset_registry import fingerprint_file, load_datasets, register_dataset


def test_registration_records_hash_and_approval(tmp_path: Path) -> None:
    dataset = tmp_path / "approved.txt"
    dataset.write_text("approved corpus", encoding="utf-8")
    registry = tmp_path / "datasets.json"

    record = register_dataset(
        registry,
        "v1",
        dataset,
        "licensed",
        "example-license",
        True,
    )

    assert record.approved is True
    assert record.sha256 == fingerprint_file(dataset)
    assert load_datasets(registry) == [record]


def test_unapproved_dataset_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "candidate.txt"
    dataset.write_text("candidate corpus", encoding="utf-8")

    with pytest.raises(ValueError, match="explicitly approved"):
        register_dataset(
            tmp_path / "datasets.json",
            "v1",
            dataset,
            "uploaded",
            "unknown",
            False,
        )


def test_registering_same_version_replaces_record(tmp_path: Path) -> None:
    dataset_one = tmp_path / "one.txt"
    dataset_two = tmp_path / "two.txt"
    dataset_one.write_text("first corpus", encoding="utf-8")
    dataset_two.write_text("second corpus", encoding="utf-8")
    registry = tmp_path / "datasets.json"

    register_dataset(registry, "v1", dataset_one, "licensed", "A", True)
    second = register_dataset(registry, "v1", dataset_two, "licensed", "B", True)

    records = load_datasets(registry)
    assert len(records) == 1
    assert records[0] == second
    assert records[0].sha256 == fingerprint_file(dataset_two)
