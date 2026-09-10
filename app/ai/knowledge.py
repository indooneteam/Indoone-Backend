from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    content: str


@dataclass(frozen=True)
class KnowledgeHit:
    document_id: str
    title: str
    content: str
    score: float


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1}


class LocalKnowledgeBase:
    """Small dependency-free retrieval layer for approved local documents."""

    def __init__(self, documents: list[KnowledgeDocument]) -> None:
        if not documents:
            raise ValueError("knowledge base needs at least one document")
        self.documents = tuple(documents)
        self._token_sets = {doc.document_id: _tokens(f"{doc.title} {doc.content}") for doc in self.documents}

    @classmethod
    def from_directory(cls, directory: Path) -> "LocalKnowledgeBase":
        documents: list[KnowledgeDocument] = []
        for path in sorted(directory.glob("*.txt")):
            content = path.read_text(encoding="utf-8").strip()
            if content:
                documents.append(
                    KnowledgeDocument(
                        document_id=path.stem,
                        title=path.stem.replace("_", " "),
                        content=content,
                    )
                )
        if not documents:
            raise ValueError(f"knowledge directory is empty: {directory}")
        return cls(documents)

    def search(self, query: str, limit: int = 3) -> list[KnowledgeHit]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        hits: list[KnowledgeHit] = []
        for document in self.documents:
            overlap = query_tokens & self._token_sets[document.document_id]
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            hits.append(
                KnowledgeHit(
                    document_id=document.document_id,
                    title=document.title,
                    content=document.content,
                    score=score,
                )
            )

        hits.sort(key=lambda item: (-item.score, item.document_id))
        return hits[:limit]


def format_hits(hits: list[KnowledgeHit]) -> str:
    if not hits:
        return ""
    parts = ["<knowledge>"]
    for hit in hits:
        parts.append(f"[{hit.title}]\n{hit.content}")
    parts.append("</knowledge>")
    return "\n".join(parts)
