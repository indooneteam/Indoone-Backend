from __future__ import annotations

import io
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader


MAX_DOCUMENT_BYTES = 8_000_000
MAX_TEXT_CHARS = 200_000
MAX_TABLES = 100
MAX_TABLE_ROWS = 500
MAX_TABLE_COLS = 100
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
    tables: list[list[list[str]]]


def _limit_text(text: str) -> tuple[str, bool]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(normalized) <= MAX_TEXT_CHARS:
        return normalized, False
    return normalized[:MAX_TEXT_CHARS], True


def _normalize_table(rows: list[list[str]]) -> list[list[str]]:
    width = min(max((len(row) for row in rows), default=0), MAX_TABLE_COLS)
    normalized: list[list[str]] = []
    for row in rows[:MAX_TABLE_ROWS]:
        cells = [" ".join(str(cell).split()) for cell in row[:width]]
        normalized.append(cells + [""] * (width - len(cells)))
    return normalized


def _validate(content: bytes) -> None:
    if not content:
        raise ValueError("document content cannot be empty")
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError("document exceeds 8 MB limit")


def _extract_pdf(filename: str, content: bytes) -> DocumentExtraction:
    reader = PdfReader(io.BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    tables: list[list[list[str]]] = []
    for page_text in pages:
        candidate_rows: list[list[str]] = []
        for line in page_text.splitlines():
            raw = line.strip()
            if raw.count("|") >= 1:
                cells = [cell.strip() for cell in raw.split("|")]
            elif "\t" in raw:
                cells = [cell.strip() for cell in raw.split("\t")]
            else:
                continue
            if len(cells) >= 2 and any(cells):
                candidate_rows.append(cells)
        if len(candidate_rows) >= 2 and len(tables) < MAX_TABLES:
            tables.append(_normalize_table(candidate_rows))

    raw_text = "\n\n".join(pages)
    text, truncated = _limit_text(raw_text)
    return DocumentExtraction(
        filename=filename,
        mime_type=PDF_MIME,
        page_count=len(reader.pages),
        paragraphs=len([p for p in text.split("\n\n") if p.strip()]),
        characters=len(text),
        text=text,
        truncated=truncated,
        tables=tables,
    )


def extract_document(filename: str, mime_type: str, content: bytes) -> DocumentExtraction:
    """Extract text and structured tables from supported documents."""

    _validate(content)
    filename = str(filename).strip()
    if not filename:
        raise ValueError("filename cannot be empty")
    mime_type = mime_type.strip().lower()
    if mime_type not in {PDF_MIME, DOCX_MIME}:
        raise ValueError("unsupported document type")

    if mime_type == PDF_MIME:
        return _extract_pdf(filename, content)

    document = Document(io.BytesIO(content))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    raw_text = "\n\n".join(paragraphs)
    text, truncated = _limit_text(raw_text)
    tables: list[list[list[str]]] = []
    for table in document.tables[:MAX_TABLES]:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        if rows:
            tables.append(_normalize_table(rows))

    return DocumentExtraction(
        filename=filename,
        mime_type=mime_type,
        page_count=None,
        paragraphs=len(paragraphs),
        characters=len(text),
        text=text,
        truncated=truncated,
        tables=tables,
    )
