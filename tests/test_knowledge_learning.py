from pathlib import Path

import pytest

from app.ai.knowledge_learning import build_entry, load_entries, save_entry


def test_build_entry_creates_deterministic_id_and_provenance() -> None:
    entry = build_entry(
        subject="Quantum Computing",
        title="Basic Concepts",
        content="Qubits can represent quantum states.",
        source_url="https://example.com/quantum",
        source_date="2026-09-12",
        learned_at="2026-09-12T12:00:00+00:00",
    )

    assert len(entry.entry_id) == 24
    assert entry.subject == "Quantum Computing"
    assert entry.source_url.startswith("https://")
    assert entry.learned_at.endswith("+00:00")


def test_save_and_load_entries(tmp_path: Path) -> None:
    entry = build_entry(subject="AI", title="Transformers", content="Attention mixes token information.")

    path = save_entry(entry, tmp_path)
    loaded = load_entries(tmp_path)

    assert path.exists()
    assert loaded == [entry]


def test_validation_rejects_empty_or_unsafe_values() -> None:
    with pytest.raises(ValueError, match="subject"):
        build_entry(subject="", title="x", content="y")
    with pytest.raises(ValueError, match="content"):
        build_entry(subject="AI", title="x", content="")
    with pytest.raises(ValueError, match="HTTP\(S\)"):
        build_entry(subject="AI", title="x", content="y", source_url="file:///tmp/x")
