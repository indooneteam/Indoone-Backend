from pathlib import Path

import pytest

import app.ai.file_context as file_context


def test_saved_file_can_be_read_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    result = file_context.save_text_file("notes.txt", "hello Indoone".encode())

    assert file_context.read_text_file(str(result["file_id"])) == "hello Indoone"


def test_invalid_file_id_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid file_id"):
        file_context.read_text_file("../outside")


def test_missing_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="not found"):
        file_context.read_text_file("00000000-0000-0000-0000-000000000000")
