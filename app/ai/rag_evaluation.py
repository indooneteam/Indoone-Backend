from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from app.ai.knowledge import LocalKnowledgeBase


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_document_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalEvaluation:
    cases: int
    hits: int
    recall_at_k: float
    mean_reciprocal_rank: float


def evaluate_retrieval(
    knowledge: LocalKnowledgeBase,
    cases: Sequence[RetrievalCase],
    *,
    limit: int = 3,
    metadata_filter: Mapping[str, str] | None = None,
) -> RetrievalEvaluation:
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if not cases:
        return RetrievalEvaluation(cases=0, hits=0, recall_at_k=0.0, mean_reciprocal_rank=0.0)

    hit_count = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        hits = knowledge.search(case.query, limit=limit, metadata_filter=metadata_filter)
        ranked_ids = [hit.document_id for hit in hits]
        if any(document_id in case.expected_document_ids for document_id in ranked_ids):
            hit_count += 1
        rank = next(
            (index for index, document_id in enumerate(ranked_ids, start=1) if document_id in case.expected_document_ids),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)

    total = float(len(cases))
    return RetrievalEvaluation(
        cases=len(cases),
        hits=hit_count,
        recall_at_k=hit_count / total,
        mean_reciprocal_rank=sum(reciprocal_ranks) / total,
    )
