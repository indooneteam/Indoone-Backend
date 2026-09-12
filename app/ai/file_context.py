from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4


FILE_ROOT = Path("data/uploads")
MAX_TEXT_BYTES = 2_000_000
SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".csv", ".log"}


def save_text_file(filename: str, content: bytes) -> dict[str, str | int]:
    if len(content) > MAX_TEXT_BYTES:
        raise ValueError("File is too large")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Text intake currently supports txt, md, json, csv, and log files")
    file_id = str(uuid4())
    target = FILE_ROOT / file_id / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    text = content.decode("utf-8", errors="replace")
    return {"file_id": file_id, "filename": target.name, "bytes": len(content), "text": text}


def read_text_file(file_id: str) -> str:
    """Read a previously uploaded text file without allowing path traversal."""
    try:
        safe_id = str(UUID(file_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid file_id") from exc

    directory = FILE_ROOT / safe_id
    matches = [path for path in directory.iterdir() if path.is_file()] if directory.exists() else []
    if len(matches) != 1:
        raise ValueError("Uploaded file was not found")
    target = matches[0]
    if target.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Stored file type is not supported")
    content = target.read_bytes()
    if len(content) > MAX_TEXT_BYTES:
        raise ValueError("Stored file is too large")
    return content.decode("utf-8", errors="replace")
