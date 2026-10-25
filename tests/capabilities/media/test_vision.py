from __future__ import annotations

import base64

import pytest

from app.capabilities.media.vision import analyze_image


PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000000020001e221bc330000000049454e44ae426082"
)


def test_image_analysis_returns_metadata_and_ocr_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.capabilities.media.vision._run_tesseract",
        lambda content: ("Hello Indoone", None),
    )

    result = analyze_image("test.png", "image/png", base64.b64encode(PNG_1X1).decode())

    assert result["filename"] == "test.png"
    assert result["width"] == 1
    assert result["height"] == 1
    assert result["ocr"]["available"] is True
    assert result["ocr"]["text"] == "Hello Indoone"
    assert result["semantic_vision"]["available"] is False


def test_image_analysis_reports_missing_ocr_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.capabilities.media.vision._run_tesseract",
        lambda content: ("", "tesseract executable is not installed"),
    )

    result = analyze_image("test.png", "image/png", base64.b64encode(PNG_1X1).decode())

    assert result["ocr"]["available"] is False
    assert "not installed" in result["ocr"]["error"]


def test_image_analysis_rejects_unsupported_mime() -> None:
    with pytest.raises(ValueError, match="unsupported image type"):
        analyze_image("test.txt", "text/plain", base64.b64encode(b"hello").decode())



def test_tesseract_uses_stdin_without_writing_image_to_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import app.capabilities.media.vision as vision

    image_bytes = b"image-bytes"
    observed: dict[str, object] = {}

    monkeypatch.setattr(vision.shutil, "which", lambda _name: "/usr/bin/tesseract")

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=b"Recognized text\n", stderr=b"")

    monkeypatch.setattr(vision.subprocess, "run", fake_run)

    text, error = vision._run_tesseract(image_bytes)

    assert (text, error) == ("Recognized text", None)
    assert observed["command"][1:3] == ["stdin", "stdout"]
    assert observed["input"] == image_bytes
    assert "text" not in observed
