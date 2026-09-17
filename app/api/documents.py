from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.api.dependencies import current_user_id
from app.capabilities.document_extract import MAX_DOCUMENT_BYTES, extract_document


router = APIRouter(prefix="/documents", tags=["documents"])
MAX_BATCH_ENCODED_BYTES = 30_000_000
MAX_DOCUMENT_BASE64_BYTES = ((MAX_DOCUMENT_BYTES + 2) // 3) * 4


class DocumentItem(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=MAX_DOCUMENT_BASE64_BYTES)


class BatchDocumentRequest(BaseModel):
    documents: list[DocumentItem] = Field(min_length=1, max_length=20)


def _decoded_size(content_base64: str) -> int | None:
    if len(content_base64) % 4:
        return None
    padding = len(content_base64) - len(content_base64.rstrip("="))
    return max(0, (len(content_base64) // 4) * 3 - padding)


def _decode_document(item: DocumentItem) -> dict[str, object]:
    decoded_size = _decoded_size(item.content_base64)
    if decoded_size is not None and decoded_size > MAX_DOCUMENT_BYTES:
        raise HTTPException(status_code=400, detail="document exceeds 8 MB limit")

    try:
        content = base64.b64decode(item.content_base64, validate=True)
        result = extract_document(item.filename, item.mime_type, content)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "filename": result.filename,
        "mime_type": result.mime_type,
        "page_count": result.page_count,
        "paragraphs": result.paragraphs,
        "characters": result.characters,
        "text": result.text,
        "truncated": result.truncated,
        "table_count": len(result.tables),
        "tables": result.tables,
        "ocr_required": not bool(result.text),
    }


@router.post("/analyze")
async def analyze_document(request: Request, body: DocumentItem) -> dict[str, object]:
    current_user_id(request)
    return {"document": _decode_document(body)}


@router.post("/analyze-batch")
async def analyze_document_batch(request: Request, body: BatchDocumentRequest) -> dict[str, object]:
    current_user_id(request)
    encoded_bytes = sum(len(item.content_base64) for item in body.documents)
    if encoded_bytes > MAX_BATCH_ENCODED_BYTES:
        raise HTTPException(status_code=413, detail="document batch is too large")
    documents = [_decode_document(item) for item in body.documents]
    return {
        "documents": documents,
        "count": len(documents),
        "total_characters": sum(int(item["characters"]) for item in documents),
        "total_tables": sum(int(item["table_count"]) for item in documents),
    }
