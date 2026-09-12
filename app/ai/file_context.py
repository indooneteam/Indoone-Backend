from __future__ import annotations

from pathlib import Path
from uuid import uuid4


FILE_ROOT = Path("data/uploads")
MAX_TEXT_BYTES = 2_000_000


def save_text_file(filename: str, content: bytes) -> dict[str, str | int]:
    if len(content) > MAX_TEXT_BYTES:
        raise ValueError("File is too large")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".json", ".csv", ".log"}:
        raise ValueError("Text intake currently supports txt, md, json, csv, and log files")
    file_id = str(uuid4())
    target = FILE_ROOT / file_id / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    text = content.decode("utf-8", errors="replace")
    return {"file_id": file_id, "filename": target.name, "bytes": len(content), "text": text}
