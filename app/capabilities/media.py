from __future__ import annotations

import base64
import os
from pathlib import Path
from uuid import uuid4


MAX_MEDIA_BYTES = 8_000_000
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def save_media(filename: str, mime_type: str, content_base64: str) -> dict[str, object]:
    filename = Path(filename).name.strip()
    mime_type = mime_type.strip().lower()
    if not filename:
        raise ValueError("filename cannot be empty")
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError("unsupported media type")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("content_base64 is invalid") from exc
    if not content:
        raise ValueError("media content cannot be empty")
    if len(content) > MAX_MEDIA_BYTES:
        raise ValueError("media exceeds 8 MB limit")

    media_id = str(uuid4())
    extension = Path(filename).suffix or ".bin"
    directory = Path(os.getenv("INDOONE_MEDIA_DIR", "data/media")) / media_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"content{extension}"
    path.write_bytes(content)
    return {
        "media_id": media_id,
        "filename": filename,
        "mime_type": mime_type,
        "bytes": len(content),
        "path": str(path),
        "vision_ready": mime_type.startswith("image/"),
        "document_ready": mime_type in {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    }
