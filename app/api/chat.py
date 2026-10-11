import base64
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.conversation_store import ConversationStore
from app.ai.file_context import read_text_file
from app.ai.service import generate_reply
from app.capabilities.document_extract import (
    DOCX_MIME,
    MAX_DOCUMENT_BYTES,
    PDF_MIME,
    extract_document,
)
from app.capabilities.vision import ALLOWED_IMAGE_MIMES, MAX_IMAGE_BYTES, analyze_image
from app.web_research.grounding import split_grounded_sources
from app.api.dependencies import current_user_id

router = APIRouter(tags=["chat"])
_store = ConversationStore()
MAX_CHAT_ATTACHMENT_BYTES = max(MAX_DOCUMENT_BYTES, MAX_IMAGE_BYTES)
MAX_CHAT_ATTACHMENT_BASE64_CHARS = ((MAX_CHAT_ATTACHMENT_BYTES + 2) // 3) * 4
CHAT_ATTACHMENT_MIME_TYPES = ALLOWED_IMAGE_MIMES | {PDF_MIME, DOCX_MIME}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    file_id: str | None = Field(default=None, min_length=1, max_length=128)
    attachment_filename: str | None = Field(default=None, min_length=1, max_length=255)
    attachment_mime_type: str | None = Field(default=None, min_length=1, max_length=120)
    attachment_base64: str | None = Field(default=None, max_length=MAX_CHAT_ATTACHMENT_BASE64_CHARS)


def _decode_chat_attachment(body: ChatRequest) -> dict[str, object] | None:
    supplied = (body.attachment_filename, body.attachment_mime_type, body.attachment_base64)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise HTTPException(status_code=400, detail="attachment filename, MIME type, and content are all required")
    if body.file_id:
        raise HTTPException(status_code=400, detail="send either file_id or a media attachment, not both")

    filename = Path(str(body.attachment_filename).replace("\\", "/")).name.strip()
    mime_type = str(body.attachment_mime_type).strip().lower()
    if not filename:
        raise HTTPException(status_code=400, detail="invalid attachment filename")
    if mime_type not in CHAT_ATTACHMENT_MIME_TYPES:
        raise HTTPException(status_code=415, detail="supported attachments are JPEG, PNG, WebP, PDF, and DOCX")

    try:
        content = base64.b64decode(str(body.attachment_base64), validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="attachment content is invalid base64") from exc
    if not content:
        raise HTTPException(status_code=400, detail="attachment content is empty")
    if len(content) > MAX_CHAT_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="attachment exceeds the 8 MB limit")

    return {"filename": filename, "mime_type": mime_type, "content": content}


class ChatSource(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    sources: list[ChatSource] = Field(default_factory=list)


def _principal(request: Request) -> str:
    return current_user_id(request)


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    user_id = _principal(request)
    is_new_conversation = body.conversation_id is None
    conversation_id = body.conversation_id or str(uuid4())

    try:
        if is_new_conversation:
            history: list[tuple[str, str]] = []
        elif _store.is_closed(conversation_id, user_id=user_id):
            conversation_id = str(uuid4())
            history = []
        else:
            # Render Free instances have an ephemeral filesystem, so the backend's
            # local SQLite history can disappear after a restart while Android still
            # has the durable Firestore conversation id. A missing local record must
            # not turn an otherwise valid chat request into HTTP 404.
            try:
                history = _store.recent(conversation_id, user_id=user_id)
            except ValueError:
                history = []
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document_context = ""
    media_attachment: dict[str, object] | None = None
    attachment = _decode_chat_attachment(body)

    if body.file_id:
        try:
            document_context = read_text_file(body.file_id, user_id=user_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if attachment is not None:
        filename = str(attachment["filename"])
        mime_type = str(attachment["mime_type"])
        content = bytes(attachment["content"])
        gemini_enabled = os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "gemini"

        if mime_type in ALLOWED_IMAGE_MIMES:
            if gemini_enabled:
                media_attachment = attachment
            else:
                # Local text inference has no semantic vision model. It can still
                # answer questions about readable text extracted from a photo.
                try:
                    vision_result = analyze_image(
                        filename,
                        mime_type,
                        base64.b64encode(content).decode("ascii"),
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                ocr_result = vision_result.get("ocr", {})
                ocr_text = str(ocr_result.get("text", "")).strip() if isinstance(ocr_result, dict) else ""
                if not ocr_text:
                    raise HTTPException(
                        status_code=503,
                        detail="This server's local text model cannot understand a photo without readable text. Enable the Gemini backend for image understanding.",
                    )
                document_context = f"Readable text extracted from image {filename}:\n{ocr_text[:100_000]}"
        elif mime_type in {PDF_MIME, DOCX_MIME}:
            try:
                document = extract_document(filename, mime_type, content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if mime_type == PDF_MIME and gemini_enabled:
                # Send the original PDF so the multimodal model can inspect charts,
                # scans, and page layout as well as any selectable text.
                media_attachment = attachment
            elif document.text.strip():
                document_context = f"Extracted content from {filename}:\n{document.text[:100_000]}"
            else:
                raise HTTPException(
                    status_code=422,
                    detail="This document contains no selectable text. Use a Gemini-enabled backend for scanned PDFs.",
                )

    try:
        if media_attachment is None:
            reply_with_sources = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
            )
        else:
            reply_with_sources = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
                media_attachment=media_attachment,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    reply, web_sources = split_grounded_sources(reply_with_sources)

    try:
        _store.append(
            conversation_id,
            [("user", body.message), ("assistant", reply_with_sources)],
            user_id=user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        sources=[ChatSource(title=source.title, url=source.url) for source in web_sources],
    )
