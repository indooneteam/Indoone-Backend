from __future__ import annotations

import base64

import pytest
from fastapi import HTTPException

from app.api.chat import ChatRequest, _decode_chat_attachment


def test_decode_chat_image_attachment() -> None:
    content = b"test-image-content"
    body = ChatRequest(
        message="Describe this image",
        attachment_filename="photo.jpg",
        attachment_mime_type="image/jpeg",
        attachment_base64=base64.b64encode(content).decode("ascii"),
    )

    result = _decode_chat_attachment(body)

    assert result == {
        "filename": "photo.jpg",
        "mime_type": "image/jpeg",
        "content": content,
    }


def test_chat_attachment_requires_all_fields() -> None:
    body = ChatRequest(
        message="Describe this image",
        attachment_filename="photo.jpg",
        attachment_mime_type="image/jpeg",
    )

    with pytest.raises(HTTPException) as error:
        _decode_chat_attachment(body)

    assert error.value.status_code == 400


def test_chat_attachment_rejects_unsupported_mime_type() -> None:
    body = ChatRequest(
        message="Summarize this video",
        attachment_filename="clip.mp4",
        attachment_mime_type="video/mp4",
        attachment_base64=base64.b64encode(b"video-content").decode("ascii"),
    )

    with pytest.raises(HTTPException) as error:
        _decode_chat_attachment(body)

    assert error.value.status_code == 415


def test_chat_attachment_rejects_invalid_base64() -> None:
    body = ChatRequest(
        message="Describe this image",
        attachment_filename="photo.jpg",
        attachment_mime_type="image/jpeg",
        attachment_base64="%%%invalid%%%",
    )

    with pytest.raises(HTTPException) as error:
        _decode_chat_attachment(body)

    assert error.value.status_code == 400


def test_chat_attachment_rejects_file_id_and_binary_attachment_together() -> None:
    body = ChatRequest(
        message="Describe this image",
        file_id="00000000-0000-0000-0000-000000000001",
        attachment_filename="photo.jpg",
        attachment_mime_type="image/jpeg",
        attachment_base64=base64.b64encode(b"image").decode("ascii"),
    )

    with pytest.raises(HTTPException) as error:
        _decode_chat_attachment(body)

    assert error.value.status_code == 400



def test_decode_multiple_chat_attachments() -> None:
    photo_a = b"first-photo"
    photo_b = b"second-photo"
    body = ChatRequest(
        message="Describe these photos",
        attachments=[
            {
                "filename": "a.jpg",
                "mime_type": "image/jpeg",
                "content_base64": base64.b64encode(photo_a).decode("ascii"),
            },
            {
                "filename": "b.png",
                "mime_type": "image/png",
                "content_base64": base64.b64encode(photo_b).decode("ascii"),
            },
        ],
    )

    from app.api.chat import _decode_chat_attachments

    decoded = _decode_chat_attachments(body)

    assert [item["filename"] for item in decoded] == ["a.jpg", "b.png"]
    assert [item["content"] for item in decoded] == [photo_a, photo_b]


def test_chat_rejects_more_than_ten_total_attachments() -> None:
    from app.api.chat import _decode_chat_attachments

    body = ChatRequest(
        message="Review these files",
        file_ids=[f"file-{i}" for i in range(10)],
        attachments=[
            {
                "filename": "extra.jpg",
                "mime_type": "image/jpeg",
                "content_base64": base64.b64encode(b"image").decode("ascii"),
            }
        ],
    )

    with pytest.raises(HTTPException) as error:
        _decode_chat_attachments(body)

    assert error.value.status_code == 400
    assert "up to 10" in error.value.detail


def test_chat_rejects_aggregate_attachment_bytes_over_limit(monkeypatch) -> None:
    from app.api import chat as chat_api

    monkeypatch.setattr(chat_api, "MAX_CHAT_TOTAL_ATTACHMENT_BYTES", 5)
    body = ChatRequest(
        message="Review these photos",
        attachments=[
            {
                "filename": "one.jpg",
                "mime_type": "image/jpeg",
                "content_base64": base64.b64encode(b"1234").decode("ascii"),
            },
            {
                "filename": "two.jpg",
                "mime_type": "image/jpeg",
                "content_base64": base64.b64encode(b"5678").decode("ascii"),
            },
        ],
    )

    with pytest.raises(HTTPException) as error:
        chat_api._decode_chat_attachments(body)

    assert error.value.status_code == 413
