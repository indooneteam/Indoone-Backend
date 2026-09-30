import json
from pathlib import Path

from scripts.self_train_and_promote import _load_training_state, _new_items, _load_collected_items


def test_new_items_excludes_already_trained_fingerprints(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "items": [
                    {"fingerprint": "a", "source":"s","title":"A","url":"https://example.com/a","summary":"A"},
                    {"fingerprint": "b", "source":"s","title":"B","url":"https://example.com/b","summary":"B"},
                ]
            }
        ),
        encoding="utf-8",
    )
    items = _load_collected_items(state)
    training_state = {
        "schema_version": 1,
        "trained_fingerprints": ["a"],
        "last_promoted_version": "old",
    }

    pending = _new_items(items, training_state)

    assert [item["fingerprint"] for item in pending] == ["b"]


def test_missing_training_state_starts_clean(tmp_path: Path) -> None:
    path = tmp_path / "training_state.json"
    state = _load_training_state(path)
    assert state["schema_version"] == 1
    assert state["trained_fingerprints"] == []
