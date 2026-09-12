from __future__ import annotations

import io

from docx import Document
from pypdf import PdfWriter

from app.capabilities.document_extract import DOCX_MIME, PDF_MIME, extract_document


def test_extract_docx_text() -> None:
    buffer = io.BytesIO()
    document = Document()
    document.add_paragraph("Indoone document understanding")
    document.add_paragraph("Kannada ಕನ್ನಡ and Hindi हिंदी")
    document.save(buffer)

    result = extract_document("sample.docx", DOCX_MIME, buffer.getvalue())

    assert result.page_count is None
    assert result.paragraphs == 2
    assert "Indoone document understanding" in result.text
    assert "ಕನ್ನಡ" in result.text
    assert result.truncated is False


def test_extract_pdf_text() -> None:
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.write(buffer)

    result = extract_document("sample.pdf", PDF_MIME, buffer.getvalue())

    assert result.page_count == 1
    assert result.mime_type == PDF_MIME
    assert result.text == ""
    assert result.truncated is False


def test_reject_unsupported_document() -> None:
    try:
        extract_document("sample.txt", "text/plain", b"hello")
    except ValueError as exc:
        assert "unsupported document type" in str(exc)
    else:
        raise AssertionError("expected unsupported document type error")
