from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
MAX_DOCUMENT_CHARS = 200_000
MAX_TOTAL_CHUNKS = 4_000
DEFAULT_CHUNK_SIZE = 1_600
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_MIN_SCORE = 0.05
DEFAULT_CONTEXT_CHARS = 12_000


@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str
    title: str
    content: str
    metadata: Mapping[str, str] | None = None


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    title: str
    content: str
    metadata: Mapping[str, str]
    ordinal: int


@dataclass(frozen=True)
class KnowledgeHit:
    document_id: str
    title: str
    content: str
    score: float
    chunk_id: str = ""
    metadata: Mapping[str, str] | None = None
    citation: str = ""
    vector_score: float = 0.0


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(text) if len(token) > 1}


def _weighted_tokens(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in _TOKEN_RE.findall(text):
        token = token.casefold()
        if len(token) <= 1:
            continue
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _embed(text: str, idf: Mapping[str, float]) -> dict[str, float]:
    counts = _weighted_tokens(text)
    if not counts:
        return {}
    vector = {token: weight * idf.get(token, 1.0) for token, weight in counts.items()}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm <= 0:
        return {}
    return {token: value / norm for token, value in vector.items()}


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _chunk_text(content: str, chunk_size: int, overlap: int) -> list[str]:
    if len(content) <= chunk_size:
        return [content]
    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(len(content), start + chunk_size)
        if end < len(content):
            boundary = content.rfind("\n", start, end)
            if boundary >= start + chunk_size // 2:
                end = boundary
        chunk = content[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(content):
            break
        start = max(start + 1, end - overlap)
    return chunks


class LocalKnowledgeBase:
    """Dependency-free RAG index for approved local documents.

    The index uses chunked TF-IDF vectors for deterministic retrieval, while the
    public score keeps query-term coverage semantics for compatibility.
    """

    def __init__(
        self,
        documents: list[KnowledgeDocument],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if not documents:
            raise ValueError("knowledge base needs at least one document")
        if chunk_size < 200:
            raise ValueError("chunk_size must be at least 200")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size-1")

        normalized_documents: list[KnowledgeDocument] = []
        for document in documents:
            content = document.content.strip()
            if not content:
                continue
            if len(content) > MAX_DOCUMENT_CHARS:
                content = content[:MAX_DOCUMENT_CHARS]
            normalized_documents.append(
                KnowledgeDocument(
                    document_id=document.document_id.strip(),
                    title=document.title.strip() or document.document_id.strip(),
                    content=content,
                    metadata=dict(document.metadata or {}),
                )
            )
        if not normalized_documents:
            raise ValueError("knowledge base needs at least one non-empty document")

        self.documents = tuple(normalized_documents)
        chunks: list[KnowledgeChunk] = []
        for document in self.documents:
            pieces = _chunk_text(document.content, chunk_size, chunk_overlap)
            for ordinal, content in enumerate(pieces):
                if len(chunks) >= MAX_TOTAL_CHUNKS:
                    break
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=f"{document.document_id}:{ordinal + 1}",
                        document_id=document.document_id,
                        title=document.title,
                        content=content,
                        metadata=dict(document.metadata or {}),
                        ordinal=ordinal,
                    )
                )
            if len(chunks) >= MAX_TOTAL_CHUNKS:
                break
        if not chunks:
            raise ValueError("knowledge base produced no chunks")
        self.chunks = tuple(chunks)

        document_frequency: dict[str, int] = {}
        for chunk in self.chunks:
            for token in _tokens(f"{chunk.title} {chunk.content}"):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = float(len(self.chunks))
        self._idf = {
            token: math.log((total + 1.0) / (frequency + 1.0)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._vectors = {
            chunk.chunk_id: _embed(f"{chunk.title} {chunk.content}", self._idf)
            for chunk in self.chunks
        }

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
                        metadata={"source": str(path), "type": "text"},
                    )
                )
        if not documents:
            raise ValueError(f"knowledge directory is empty: {directory}")
        return cls(documents)

    def search(
        self,
        query: str,
        limit: int = 3,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        metadata_filter: Mapping[str, str] | None = None,
    ) -> list[KnowledgeHit]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if min_score < 0 or min_score > 1:
            raise ValueError("min_score must be between zero and one")

        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        query_vector = _embed(query, self._idf)
        hits: list[KnowledgeHit] = []
        for chunk in self.chunks:
            if metadata_filter and any(chunk.metadata.get(key) != value for key, value in metadata_filter.items()):
                continue
            overlap = query_tokens & _tokens(f"{chunk.title} {chunk.content}")
            if not overlap:
                continue
            coverage = len(overlap) / len(query_tokens)
            vector_score = _cosine(query_vector, self._vectors[chunk.chunk_id])
            score = max(0.0, min(1.0, coverage))
            if score < min_score:
                continue
            hits.append(
                KnowledgeHit(
                    document_id=chunk.document_id,
                    title=chunk.title,
                    content=chunk.content,
                    score=score,
                    chunk_id=chunk.chunk_id,
                    metadata=dict(chunk.metadata),
                    citation=f"[{chunk.document_id}:{chunk.ordinal + 1}]",
                    vector_score=vector_score,
                )
            )

        hits.sort(key=lambda item: (-item.score, -item.vector_score, item.document_id, item.chunk_id))
        return hits[:limit]


def build_context(hits: list[KnowledgeHit], max_chars: int = DEFAULT_CONTEXT_CHARS) -> str:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")
    parts = ["<knowledge>"]
    used = len(parts[0]) + 1
    for hit in hits:
        block = f"{hit.citation or '[' + hit.document_id + ']'} {hit.title}\n{hit.content}"
        if used + len(block) + 1 > max_chars:
            break
        parts.append(block)
        used += len(block) + 1
    parts.append("</knowledge>")
    return "\n".join(parts) if len(parts) > 2 else ""


def format_hits(hits: list[KnowledgeHit]) -> str:
    return build_context(hits)
