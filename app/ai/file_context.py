from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4


FILE_ROOT = Path("data/uploads")
MAX_TEXT_BYTES = 2_000_000
SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".csv", ".log"}


def _safe_directory(file_id: str) -> Path:
    try:
        safe_id = str(UUID(file_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("Invalid file_id") from exc
    return FILE_ROOT / safe_id


def _load_metadata(directory: Path) -> dict[str, object]:
    metadata_path = directory / ".metadata.json"
    if not metadata_path.exists():
        raise PermissionError("uploaded file owner metadata is missing")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Uploaded file metadata is invalid") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Uploaded file metadata is invalid")
    return metadata


def save_text_file(filename: str, content: bytes, user_id: str = "") -> dict[str, str | int]:
    if len(content) > MAX_TEXT_BYTES:
        raise ValueError("File is too large")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Text intake currently supports txt, md, json, csv, and log files")
    owner = user_id.strip()
    if not owner or len(owner) > 256:
        raise ValueError("authenticated user is required")
    file_id = str(uuid4())
    directory = FILE_ROOT / file_id
    target = directory / Path(filename).name
    directory.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    metadata = {"user_id": owner, "filename": target.name}
    (directory / ".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    text = content.decode("utf-8", errors="replace")
    return {"file_id": file_id, "filename": target.name, "bytes": len(content), "text": text}


def get_file_owner(file_id: str) -> str:
    metadata = _load_metadata(_safe_directory(file_id))
    owner = str(metadata.get("user_id", "")).strip()
    if not owner:
        raise PermissionError("uploaded file owner metadata is missing")
    return owner


def read_text_file(file_id: str, user_id: str = "") -> str:
    directory = _safe_directory(file_id)
    owner = user_id.strip()
    if not owner:
        raise PermissionError("authenticated user required")
    stored_owner = get_file_owner(file_id)
    if stored_owner != owner:
        raise PermissionError("file belongs to another user")

    matches = [path for path in directory.iterdir() if path.is_file() and path.name != ".metadata.json"] if directory.exists() else []
    if len(matches) != 1:
        raise ValueError("Uploaded file was not found")
    target = matches[0]
    if target.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError("Stored file type is not supported")
    content = target.read_bytes()
    if len(content) > MAX_TEXT_BYTES:
        raise ValueError("Stored file is too large")
    return content.decode("utf-8", errors="replace")
