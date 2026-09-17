import pytest

from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase
from app.ai.rag_evaluation import RetrievalCase, evaluate_retrieval


def test_retrieval_evaluation_reports_recall_and_mrr() -> None:
    knowledge = LocalKnowledgeBase(
        [
            KnowledgeDocument("ai", "AI", "python training and inference"),
            KnowledgeDocument("ops", "Operations", "deployment and incident response"),
        ]
    )

    evaluation = evaluate_retrieval(
        knowledge,
        [
            RetrievalCase("python inference", frozenset({"ai"})),
            RetrievalCase("incident response", frozenset({"ops"})),
        ],
        limit=1,
    )

    assert evaluation.cases == 2
    assert evaluation.hits == 2
    assert evaluation.recall_at_k == 1.0
    assert evaluation.mean_reciprocal_rank == 1.0


def test_retrieval_evaluation_validates_limit() -> None:
    knowledge = LocalKnowledgeBase([KnowledgeDocument("x", "X", "hello world")])
    with pytest.raises(ValueError, match="limit"):
        evaluate_retrieval(knowledge, [RetrievalCase("hello", frozenset({"x"}))], limit=0)
