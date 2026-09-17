from __future__ import annotations

import base64

import pytest

import app.api.documents as documents
from app.api.documents import DocumentItem, _decode_document



def test_decode_document_rejects_oversized_base64_before_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents, "MAX_DOCUMENT_BYTES", 4)
    item = DocumentItem(
        filename="sample.pdf",
        mime_type="application/pdf",
        content_base64=base64.b64encode(b"12345").decode("ascii"),
    )

    monkeypatch.setattr(documents.base64, "b64decode", lambda *args, **kwargs: pytest.fail("decode should not run"))

    with pytest.raises(Exception) as exc_info:
        _decode_document(item)

    assert getattr(exc_info.value, "status_code", None) == 400
    assert "document exceeds 8 MB limit" in str(getattr(exc_info.value, "detail", ""))


def test_decode_document_allows_payload_at_decoded_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(documents, "MAX_DOCUMENT_BYTES", 4)
    item = DocumentItem(
        filename="sample.bin",
        mime_type="application/octet-stream",
        content_base64=base64.b64encode(b"1234").decode("ascii"),
    )

    with pytest.raises(Exception) as exc_info:
        _decode_document(item)

    assert getattr(exc_info.value, "status_code", None) == 400
    assert "unsupported document type" in str(getattr(exc_info.value, "detail", ""))
