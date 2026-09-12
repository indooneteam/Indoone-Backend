from pathlib import Path

from scripts.assemble_training_dataset import assemble


def test_assemble_removes_duplicates_and_splits(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"instruction":"A","response":"one","category":"reasoning"}\n'
        '{"instruction":"A","response":"one","category":"reasoning"}\n'
        '{"instruction":"B","response":"two","category":"math"}\n'
        '{"instruction":"C","response":"three","category":"coding"}\n'
        '{"instruction":"D","response":"four","category":"translation"}\n',
        encoding="utf-8",
    )
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"

    report = assemble([source], train, validation, validation_ratio=0.25, seed=42)

    assert report["total"] == 4
    assert report["duplicates_removed"] == 1
    assert report["train"] == 3
    assert report["validation"] == 1
    assert len(train.read_text(encoding="utf-8").splitlines()) == 3
    assert len(validation.read_text(encoding="utf-8").splitlines()) == 1


def test_assemble_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"instruction":"A","response":"one","category":"reasoning"}\n'
        '{"instruction":"B","response":"two","category":"math"}\n'
        '{"instruction":"C","response":"three","category":"coding"}\n'
        '{"instruction":"D","response":"four","category":"translation"}\n',
        encoding="utf-8",
    )
    train_a = tmp_path / "a.jsonl"
    validation_a = tmp_path / "av.jsonl"
    train_b = tmp_path / "b.jsonl"
    validation_b = tmp_path / "bv.jsonl"

    assemble([source], train_a, validation_a, validation_ratio=0.25, seed=42)
    assemble([source], train_b, validation_b, validation_ratio=0.25, seed=42)

    assert train_a.read_text(encoding="utf-8") == train_b.read_text(encoding="utf-8")
    assert validation_a.read_text(encoding="utf-8") == validation_b.read_text(encoding="utf-8")
