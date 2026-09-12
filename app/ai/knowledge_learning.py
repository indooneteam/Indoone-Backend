"""Approved long-term subject knowledge storage for Indoone.

This module is intentionally separate from live web research. Fresh web results
are not persisted automatically. Only explicitly approved/curated knowledge
records are serialized for the long-term knowledge corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


MAX_SUBJECT_LENGTH = 200
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 200_000
MAX_SOURCE_URL_LENGTH = 2_000


class KnowledgeBackup(Protocol):
    """Minimal interface required for durable object-storage backups."""

    def upload_file(self, local_path: Path, object_key: str) -> None: ...


@dataclass(frozen=True)
class KnowledgeEntry:
    """A durable, provenance-aware piece of approved subject knowledge."""

    entry_id: str
    subject: str
    title: str
    content: str
    source_url: str = ""
    source_date: str = ""
    learned_at: str = ""


def _clean(value: str, maximum: int) -> str:
    return " ".join(value.strip().split())[:maximum]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_entry(
    *,
    subject: str,
    title: str,
    content: str,
    source_url: str = "",
    source_date: str = "",
    learned_at: str | None = None,
) -> KnowledgeEntry:
    """Validate and build an approved knowledge entry."""

    subject = _clean(subject, MAX_SUBJECT_LENGTH)
    title = _clean(title, MAX_TITLE_LENGTH)
    content = content.strip()[:MAX_CONTENT_LENGTH]
    source_url = source_url.strip()[:MAX_SOURCE_URL_LENGTH]
    source_date = _clean(source_date, 50)

    if not subject:
        raise ValueError("subject cannot be empty")
    if not title:
        raise ValueError("title cannot be empty")
    if not content:
        raise ValueError("content cannot be empty")
    if source_url and not source_url.startswith(("http://", "https://")):
        raise ValueError("source_url must be HTTP(S)")

    digest = hashlib.sha256(
        f"{subject}\n{title}\n{content}\n{source_url}\n{source_date}".encode("utf-8")
    ).hexdigest()[:24]

    return KnowledgeEntry(
        entry_id=digest,
        subject=subject,
        title=title,
        content=content,
        source_url=source_url,
        source_date=source_date,
        learned_at=learned_at or _utc_now(),
    )


def save_entry(
    entry: KnowledgeEntry,
    directory: Path,
    backup: KnowledgeBackup | None = None,
    backup_prefix: str = "knowledge/",
) -> Path:
    """Persist one approved entry locally and optionally back it up to B2."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{entry.entry_id}.json"
    path.write_text(
        json.dumps(asdict(entry), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if backup is not None:
        prefix = backup_prefix.strip("/")
        object_key = f"{prefix}/{path.name}" if prefix else path.name
        backup.upload_file(path, object_key)
    return path


def load_entries(directory: Path) -> list[KnowledgeEntry]:
    """Load valid JSON knowledge entries from a directory."""

    entries: list[KnowledgeEntry] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries.append(KnowledgeEntry(**payload))
    return entries
