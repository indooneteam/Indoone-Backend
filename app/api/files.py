from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.file_context import MAX_TEXT_BYTES, save_text_file
from app.api.dependencies import current_user_id

router = APIRouter(tags=["files"])

MAX_FILE_BASE64_BYTES = ((MAX_TEXT_BYTES + 2) // 3) * 4


class FileUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=MAX_FILE_BASE64_BYTES)


def _decoded_size(content_base64: str) -> int | None:
    if len(content_base64) % 4:
        return None
    padding = len(content_base64) - len(content_base64.rstrip("="))
    return max(0, (len(content_base64) // 4) * 3 - padding)


@router.post("/files")
async def upload_file(request: Request, body: FileUploadRequest) -> dict[str, object]:
    user_id = current_user_id(request)
    decoded_size = _decoded_size(body.content_base64)
    if decoded_size is not None and decoded_size > MAX_TEXT_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds 2 MB limit")

    try:
        content = base64.b64decode(body.content_base64, validate=True)
        result = save_text_file(body.filename, content, user_id=user_id)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "file_id": str(result["file_id"]),
        "filename": str(result["filename"]),
        "bytes": int(result["bytes"]),
        "text_preview": str(result["text"])[:500],
    }
