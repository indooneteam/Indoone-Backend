from __future__ import annotations

import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

MAX_IMAGE_BYTES = 8_000_000
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}


def _image_dimensions(content: bytes, mime_type: str) -> tuple[int | None, int | None]:
    try:
        if mime_type == "image/png" and content[:8] == b"\x89PNG\r\n\x1a\n":
            return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")
        if mime_type == "image/jpeg" and content[:2] == b"\xff\xd8":
            stream = io.BytesIO(content)
            stream.read(2)
            while True:
                marker_start = stream.read(1)
                if not marker_start:
                    break
                if marker_start != b"\xff":
                    continue
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                if marker in {b"\xd8", b"\xd9"}:
                    continue
                size_bytes = stream.read(2)
                if len(size_bytes) != 2:
                    break
                size = int.from_bytes(size_bytes, "big")
                if marker[0] in range(0xC0, 0xC4):
                    stream.read(1)
                    height = int.from_bytes(stream.read(2), "big")
                    width = int.from_bytes(stream.read(2), "big")
                    return width, height
                stream.seek(size - 2, io.SEEK_CUR)
    except (IndexError, OSError, ValueError):
        pass
    return None, None


def _run_tesseract(content: bytes, suffix: str) -> tuple[str, str | None]:
    executable = shutil.which("tesseract")
    if executable is None:
        return "", "tesseract executable is not installed"
    with tempfile.TemporaryDirectory() as directory:
        image_path = Path(directory) / f"input{suffix}"
        image_path.write_bytes(content)
        output_base = Path(directory) / "ocr"
        try:
            completed = subprocess.run(
                [executable, str(image_path), str(output_base), "--psm", "6"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return "", f"tesseract failed: {exc}"
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown OCR error"
            return "", detail
        text_path = output_base.with_suffix(".txt")
        if not text_path.exists():
            return "", "tesseract produced no text output"
        return text_path.read_text(encoding="utf-8", errors="replace").strip(), None


def analyze_image(filename: str, mime_type: str, content_base64: str) -> dict[str, object]:
    filename = Path(filename).name.strip()
    mime_type = mime_type.strip().lower()
    if not filename:
        raise ValueError("filename cannot be empty")
    if mime_type not in ALLOWED_IMAGE_MIMES:
        raise ValueError("unsupported image type")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("content_base64 is invalid") from exc
    if not content:
        raise ValueError("image content cannot be empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 8 MB limit")

    width, height = _image_dimensions(content, mime_type)
    suffix = Path(filename).suffix or ".img"
    text, ocr_error = _run_tesseract(content, suffix)
    return {
        "filename": filename,
        "mime_type": mime_type,
        "bytes": len(content),
        "width": width,
        "height": height,
        "ocr": {
            "available": ocr_error is None,
            "text": text,
            "characters": len(text),
            "error": ocr_error,
        },
        "semantic_vision": {
            "available": False,
            "reason": "A dedicated local vision model is not installed yet; OCR and image metadata are available.",
        },
    }
