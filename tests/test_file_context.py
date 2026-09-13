from pathlib import Path

import pytest

import app.ai.file_context as file_context


OWNER = "user-one"


def test_saved_file_can_be_read_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    result = file_context.save_text_file("notes.txt", "hello Indoone".encode(), user_id=OWNER)

    assert file_context.read_text_file(str(result["file_id"]), user_id=OWNER) == "hello Indoone"


def test_invalid_file_id_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid file_id"):
        file_context.read_text_file("../outside", user_id=OWNER)


def test_missing_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="not found"):
        file_context.read_text_file("00000000-0000-0000-0000-000000000000", user_id=OWNER)


def test_unsupported_upload_suffix_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="supports txt, md, json, csv, and log"):
        file_context.save_text_file("payload.exe", b"hello", user_id=OWNER)


def test_upload_size_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="too large"):
        file_context.save_text_file(
            "large.txt",
            b"x" * (file_context.MAX_TEXT_BYTES + 1),
            user_id=OWNER,
        )


def test_stored_file_size_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    result = file_context.save_text_file("notes.txt", b"hello", user_id=OWNER)
    target = tmp_path / result["file_id"] / "notes.txt"
    target.write_bytes(b"x" * (file_context.MAX_TEXT_BYTES + 1))

    with pytest.raises(ValueError, match="Stored file is too large"):
        file_context.read_text_file(str(result["file_id"]), user_id=OWNER)
