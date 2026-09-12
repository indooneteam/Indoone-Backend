from pathlib import Path

import pytest

from scripts.audit_dataset import audit_dataset


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(__import__("json").dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def test_audit_reports_small_dataset_as_not_ready(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    instructions = tmp_path / "instructions.jsonl"
    multilingual = tmp_path / "multilingual.jsonl"
    corpus.write_text("small dataset\n", encoding="utf-8")
    _write_jsonl(instructions, [{"instruction": "Hi", "response": "Hello", "category": "conversation"}])
    _write_jsonl(multilingual, [{"instruction": "ನಮಸ್ಕಾರ", "response": "ಹಲೋ", "category": "language"}])

    report = audit_dataset(
        corpus,
        [instructions],
        multilingual,
        min_corpus_chars=100,
        min_instruction_examples=10,
        min_multilingual_examples=5,
    )

    assert report["ready"] is False
    assert report["corpus_characters"] == len("small dataset\n")
    assert report["instruction_examples"] == 1
    assert report["multilingual_examples"] == 1
    assert report["failures"]


def test_audit_passes_clean_dataset(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    instructions = tmp_path / "instructions.jsonl"
    multilingual = tmp_path / "multilingual.jsonl"
    corpus.write_text("a" * 200, encoding="utf-8")
    rows = [
        {"instruction": f"Question {i}", "response": f"Answer {i}", "category": "general"}
        for i in range(10)
    ]
    corpus.write_text("a" * 200, encoding="utf-8")
    _write_jsonl(instructions, rows)
    _write_jsonl(
        multilingual,
        [
            {"instruction": "ನಮಸ್ಕಾರ", "response": "ಹಲೋ", "category": "kannada"},
            {"instruction": "हेलो", "response": "नमस्ते", "category": "hindi"},
        ]
    )

    report = audit_dataset(
        corpus,
        [instructions],
        multilingual,
        min_corpus_chars=100,
        min_instruction_examples=10,
        min_multilingual_examples=2,
        max_duplicate_rate=0.0,
    )

    assert report["ready"] is True
    assert report["duplicate_rate"] == 0.0


def test_audit_rejects_duplicate_instruction_pairs(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    instructions = tmp_path / "instructions.jsonl"
    multilingual = tmp_path / "multilingual.jsonl"
    corpus.write_text("a" * 100, encoding="utf-8")
    duplicate = {"instruction": "same", "response": "same", "category": "general"}
    _write_jsonl(instructions, [duplicate, duplicate])
    _write_jsonl(multilingual, [{"instruction": "x", "response": "y", "category": "language"}])

    report = audit_dataset(
        corpus,
        [instructions],
        multilingual,
        min_corpus_chars=100,
        min_instruction_examples=2,
        min_multilingual_examples=1,
        max_duplicate_rate=0.0,
    )

    assert report["ready"] is False
    assert report["duplicate_rate"] == 0.5
    assert any("duplicate rate" in failure for failure in report["failures"])
