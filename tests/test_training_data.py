import json
from pathlib import Path

import pytest

from app.ai.training_data import TrainingExample, load_examples, write_corpus


def test_load_examples_validates_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"instruction": " Hello world ", "response": " A response ", "category": "general"}),
                json.dumps({"instruction": "hello world", "response": "a response", "category": "general"}),
            ]
        ),
        encoding="utf-8",
    )

    examples = load_examples(source)

    assert examples == [TrainingExample("Hello world", "A response", "general")]


def test_load_examples_rejects_invalid_data(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(json.dumps({"instruction": "", "response": "ok"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="instruction"):
        load_examples(source)


def test_write_corpus_is_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "corpus.txt"
    examples = [TrainingExample("Question", "Answer", "general")]

    written = write_corpus(examples, output)

    assert written == output
    assert output.read_text(encoding="utf-8") == (
        "<instruction>\nQuestion\n</instruction>\n"
        "<response>\nAnswer\n</response>\n"
    )
