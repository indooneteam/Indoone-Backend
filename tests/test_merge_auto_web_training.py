from pathlib import Path
import json

from scripts.merge_auto_web_training import build_examples, merge_into_training_file


def _state(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "items": [
                    {
                        "source": "NASA News",
                        "title": "Example update",
                        "url": "https://www.nasa.gov/example",
                        "summary": "A trusted summary.",
                        "published": "2026-09-30T00:00:00Z",
                        "fingerprint": "abc123",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_examples_adds_three_source_attributed_variants(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    _state(state)
    examples = build_examples(state)

    assert len(examples) == 3
    assert all(item["category"] == "web_update" for item in examples)
    assert all("Source: NASA News" in item["response"] for item in examples)
    assert all("https://www.nasa.gov/example" in item["response"] for item in examples)


def test_merge_is_idempotent(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    output = tmp_path / "generated.jsonl"
    _state(state)

    first = merge_into_training_file(state_path=state, output_path=output)
    second = merge_into_training_file(state_path=state, output_path=output)

    assert first == 3
    assert second == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
