import asyncio
import base64
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.api.files as files_api


OWNER = "user-one"


def _request(user_id: str = OWNER):
    return SimpleNamespace(state=SimpleNamespace(principal_id=user_id))


def test_upload_file_saves_authenticated_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_save_text_file(filename, content, user_id=""):
        captured.update(filename=filename, content=content, user_id=user_id)
        return {
            "file_id": "00000000-0000-0000-0000-000000000001",
            "filename": filename,
            "bytes": len(content),
            "text": "hello Indoone",
        }

    monkeypatch.setattr(files_api, "save_text_file", fake_save_text_file)
    body = files_api.FileUploadRequest(
        filename="notes.txt",
        content_base64=base64.b64encode(b"hello Indoone").decode(),
    )

    result = asyncio.run(files_api.upload_file(_request(), body))

    assert captured == {
        "filename": "notes.txt",
        "content": b"hello Indoone",
        "user_id": OWNER,
    }
    assert result["file_id"] == "00000000-0000-0000-0000-000000000001"
    assert result["filename"] == "notes.txt"
    assert result["bytes"] == 13
    assert result["text_preview"] == "hello Indoone"


def test_upload_file_rejects_invalid_base64() -> None:
    body = files_api.FileUploadRequest(filename="notes.txt", content_base64="not-base64")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(files_api.upload_file(_request(), body))

    assert exc.value.status_code == 400


def test_upload_file_rejects_decoded_payload_above_limit() -> None:
    oversized = b"x" * (files_api.MAX_TEXT_BYTES + 1)
    body = files_api.FileUploadRequest(
        filename="notes.txt",
        content_base64=base64.b64encode(oversized).decode(),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(files_api.upload_file(_request(), body))

    assert exc.value.status_code == 413


def test_upload_file_requires_authentication() -> None:
    body = files_api.FileUploadRequest(
        filename="notes.txt",
        content_base64=base64.b64encode(b"hello").decode(),
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(files_api.upload_file(_request(""), body))

    assert exc.value.status_code == 401
