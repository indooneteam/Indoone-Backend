from pathlib import Path

import pytest

from app.ai import file_context


def test_uploaded_file_requires_matching_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")

    assert file_context.read_text_file(saved["file_id"], user_id="user-one") == "hello"
    with pytest.raises(PermissionError, match="another user"):
        file_context.read_text_file(saved["file_id"], user_id="user-two")


def test_owned_file_requires_authentication(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")

    with pytest.raises(PermissionError, match="authenticated user required"):
        file_context.read_text_file(saved["file_id"])
