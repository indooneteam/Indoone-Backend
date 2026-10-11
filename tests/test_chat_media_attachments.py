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
