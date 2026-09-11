from pathlib import Path

import pytest

from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase, format_hits


def test_local_knowledge_search_ranks_relevant_documents() -> None:
    knowledge = LocalKnowledgeBase(
        [
            KnowledgeDocument("ai", "Indoone AI", "local model and training pipeline"),
            KnowledgeDocument("memory", "Conversation Memory", "recent messages provide context"),
        ]
    )

    hits = knowledge.search("model training", limit=2)

    assert hits[0].document_id == "ai"
    assert hits[0].score == 1.0


def test_search_and_format_are_validated() -> None:
    knowledge = LocalKnowledgeBase([KnowledgeDocument("x", "Test", "hello world")])

    with pytest.raises(ValueError, match="query"):
        knowledge.search("   ")
    with pytest.raises(ValueError, match="limit"):
        knowledge.search("hello", limit=0)
    assert format_hits([]) == ""


def test_directory_loader_reads_text_documents(tmp_path: Path) -> None:
    (tmp_path / "first_doc.txt").write_text("Local knowledge", encoding="utf-8")
    (tmp_path / "second_doc.txt").write_text("Another document", encoding="utf-8")

    knowledge = LocalKnowledgeBase.from_directory(tmp_path)

    assert [doc.document_id for doc in knowledge.documents] == ["first_doc", "second_doc"]
