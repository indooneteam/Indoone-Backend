from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.document_extract import extract_document

router = APIRouter(tags=["documents"])


class DocumentItem(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class BatchDocumentRequest(BaseModel):
    documents: list[DocumentItem] = Field(min_length=1, max_length=10)


def _decode_and_extract(request: DocumentItem) -> dict[str, Any]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        result = extract_document(request.filename, request.mime_type, content)
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
        "tables": result.tables,
        "table_count": len(result.tables),
        "ocr_required": not bool(result.text),
    }


@router.post("/documents/analyze")
async def analyze_document(request: DocumentItem) -> dict[str, Any]:
    return {"document": _decode_and_extract(request)}


@router.post("/documents/analyze-batch")
async def analyze_documents(request: BatchDocumentRequest) -> dict[str, Any]:
    documents = [_decode_and_extract(item) for item in request.documents]
    return {"documents": documents, "count": len(documents)}
