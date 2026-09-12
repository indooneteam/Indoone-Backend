from __future__ import annotations

import io
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader


MAX_DOCUMENT_BYTES = 8_000_000
MAX_TEXT_CHARS = 200_000
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass(frozen=True)
class DocumentExtraction:
    filename: str
    mime_type: str
    page_count: int | None
    paragraphs: int
    characters: int
    text: str
    truncated: bool


def _limit_text(text: str) -> tuple[str, bool]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= MAX_TEXT_CHARS:
        return normalized, False
    return normalized[:MAX_TEXT_CHARS], True


def _validate(content: bytes) -> None:
    if not content:
        raise ValueError("document content cannot be empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("document exceeds 8 MB limit")


def extract_document(filename: str, mime_type: str, content: bytes) -> DocumentExtraction:
    """Extract machine-readable text from PDF or DOCX files.

    This first implementation is text extraction only. Scanned PDFs that contain
    images without an embedded text layer require OCR/vision in a later stage.
    """

    _validate(content)
    mime_type = mime_type.strip().lower()
    if mime_type not in {PDF_MIME, DOCX_MIME}:
        raise ValueError("unsupported document type")

    if mime_type == PDF_MIME:
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        raw_text = "\n\n".join(pages)
        text, truncated = _limit_text(raw_text)
        return DocumentExtraction(
            filename=filename,
            mime_type=mime_type,
            page_count=len(reader.pages),
            paragraphs=len([p for p in text.split("\n\n") if p.strip()]),
            characters=len(text),
            text=text,
            truncated=truncated,
        )

    document = Document(io.BytesIO(content))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    raw_text = "\n\n".join(paragraphs)
    text, truncated = _limit_text(raw_text)
    return DocumentExtraction(
        filename=filename,
        mime_type=mime_type,
        page_count=None,
        paragraphs=len(paragraphs),
        characters=len(text),
        text=text,
        truncated=truncated,
    )
