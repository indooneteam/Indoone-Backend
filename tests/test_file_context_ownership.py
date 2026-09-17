from pathlib import Path

import pytest

from app.ai import file_context


def test_uploaded_file_requires_matching_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")

    assert file_context.get_file_owner(saved["file_id"]) == "user-one"
    assert file_context.read_text_file(saved["file_id"], user_id="user-one") == "hello"
    with pytest.raises(PermissionError, match="another user"):
        file_context.read_text_file(saved["file_id"], user_id="user-two")


def test_owned_file_requires_authentication(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")

    with pytest.raises(PermissionError, match="authenticated user required"):
        file_context.read_text_file(saved["file_id"])


def test_upload_requires_authenticated_owner(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="authenticated user is required"):
        file_context.save_text_file("note.txt", b"hello")


def test_missing_owner_metadata_is_not_usable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")
    (tmp_path / saved["file_id"] / ".metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PermissionError, match="owner metadata is missing"):
        file_context.get_file_owner(saved["file_id"])


def test_path_traversal_filename_is_normalized_to_basename(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("../../secret.txt", b"hello", user_id="user-one")

    assert saved["filename"] == "secret.txt"
    assert (tmp_path / saved["file_id"] / "secret.txt").exists()


def test_filename_length_is_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Invalid filename"):
        file_context.save_text_file("a" * 256 + ".txt", b"hello", user_id="user-one")


def test_tampered_filename_metadata_cannot_select_another_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(file_context, "FILE_ROOT", tmp_path)
    saved = file_context.save_text_file("note.txt", b"hello", user_id="user-one")
    directory = tmp_path / saved["file_id"]
    (directory / ".metadata.json").write_text(
        '{"user_id":"user-one","filename":"other.txt"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not found"):
        file_context.read_text_file(saved["file_id"], user_id="user-one")
