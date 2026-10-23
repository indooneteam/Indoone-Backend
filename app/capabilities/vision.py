from __future__ import annotations

import base64
import io
import shutil
import subprocess
from pathlib import Path

MAX_IMAGE_BYTES = 5_000_000
# The encoded length has a known relationship to the decoded length, so an oversize
# payload can be refused before it is decoded rather than after.
MAX_IMAGE_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
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


def _run_tesseract(content: bytes) -> tuple[str, str | None]:
    """Run OCR through stdin/stdout so the uploaded image is not written to disk."""

    executable = shutil.which("tesseract")
    if executable is None:
        return "", "tesseract executable is not installed"
    try:
        completed = subprocess.run(
            [executable, "stdin", "stdout", "--psm", "6"],
            input=content,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", f"tesseract failed: {exc}"
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip() or "unknown OCR error"
        return "", detail
    return completed.stdout.decode("utf-8", errors="replace").strip(), None

def _require_image_signature(content: bytes, mime_type: str) -> None:
    """Refuse bytes that are not the image the caller says they are.

    The declared type decides which branch handles the upload and which size limit
    applies, so it must be matched against the content, not trusted.
    """
    if mime_type == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image content does not match the declared type")
    if mime_type == "image/jpeg" and not content.startswith(b"\xff\xd8\xff"):
        raise ValueError("image content does not match the declared type")
    if mime_type == "image/webp" and not (content[:4] == b"RIFF" and content[8:12] == b"WEBP"):
        raise ValueError("image content does not match the declared type")


def analyze_image(filename: str, mime_type: str, content_base64: str) -> dict[str, object]:
    filename = Path(filename).name.strip()
    mime_type = mime_type.strip().lower()
    if not filename:
        raise ValueError("filename cannot be empty")
    if mime_type not in ALLOWED_IMAGE_MIMES:
        raise ValueError("unsupported image type")
    if len(content_base64) > MAX_IMAGE_BASE64_CHARS:
        raise ValueError("image exceeds 5 MB limit")
    try:
        content = base64.b64decode(content_base64, validate=True)
    except Exception as exc:
        raise ValueError("content_base64 is invalid") from exc
    if not content:
        raise ValueError("image content cannot be empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 5 MB limit")
    _require_image_signature(content, mime_type)

    width, height = _image_dimensions(content, mime_type)
    text, ocr_error = _run_tesseract(content)
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
