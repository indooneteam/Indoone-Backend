from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.document_extract import extract_document


router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentItem(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class BatchDocumentRequest(BaseModel):
    documents: list[DocumentItem] = Field(min_length=1, max_length=20)


def _decode_document(item: DocumentItem) -> dict[str, object]:
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
async def analyze_document(request: DocumentItem) -> dict[str, object]:
    return {"document": _decode_document(request)}


@router.post("/analyze-batch")
async def analyze_document_batch(request: BatchDocumentRequest) -> dict[str, object]:
    documents = [_decode_document(item) for item in request.documents]
    return {
        "documents": documents,
        "count": len(documents),
        "total_characters": sum(int(item["characters"]) for item in documents),
        "total_tables": sum(int(item["table_count"]) for item in documents),
    }
