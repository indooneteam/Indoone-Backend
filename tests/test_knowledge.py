from pathlib import Path

import pytest

from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase, build_context, format_hits


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
    assert hits[0].chunk_id == "ai:1"
    assert hits[0].citation == "[ai:1]"
    assert hits[0].vector_score > 0


def test_search_and_format_are_validated() -> None:
    knowledge = LocalKnowledgeBase([KnowledgeDocument("x", "Test", "hello world")])

    with pytest.raises(ValueError, match="query"):
        knowledge.search("   ")
    with pytest.raises(ValueError, match="limit"):
        knowledge.search("hello", limit=0)
    with pytest.raises(ValueError, match="min_score"):
        knowledge.search("hello", min_score=2)
    with pytest.raises(ValueError, match="max_chars"):
        build_context([], max_chars=0)
    assert format_hits([]) == ""


def test_directory_loader_reads_text_documents(tmp_path: Path) -> None:
    (tmp_path / "first_doc.txt").write_text("Local knowledge", encoding="utf-8")
    (tmp_path / "second_doc.txt").write_text("Another document", encoding="utf-8")

    knowledge = LocalKnowledgeBase.from_directory(tmp_path)

    assert [doc.document_id for doc in knowledge.documents] == ["first_doc", "second_doc"]
    assert knowledge.documents[0].metadata["type"] == "text"


def test_long_documents_are_chunked_and_context_is_bounded() -> None:
    content = ("alpha beta gamma\n" * 300).strip()
    knowledge = LocalKnowledgeBase([KnowledgeDocument("long", "Long document", content)], chunk_size=300, chunk_overlap=50)

    assert len(knowledge.chunks) > 1
    hits = knowledge.search("alpha beta", limit=10)
    context = build_context(hits, max_chars=600)

    assert hits
    assert len(context) <= 600
    assert "[long:" in context


def test_metadata_filter_and_relevance_threshold() -> None:
    knowledge = LocalKnowledgeBase(
        [
            KnowledgeDocument("one", "Engineering", "python deployment guide", metadata={"team": "eng"}),
            KnowledgeDocument("two", "Sales", "python sales guide", metadata={"team": "sales"}),
        ]
    )

    hits = knowledge.search("python deployment", metadata_filter={"team": "eng"})
    assert hits and {hit.document_id for hit in hits} == {"one"}

    assert knowledge.search("python", min_score=0.9)
    assert knowledge.search("missing-term") == []


def test_large_documents_and_chunk_counts_are_bounded() -> None:
    from app.ai.knowledge import MAX_DOCUMENT_CHARS, MAX_TOTAL_CHUNKS

    oversized = "knowledge " * (MAX_DOCUMENT_CHARS // 10 + 1)
    knowledge = LocalKnowledgeBase([KnowledgeDocument("huge", "Huge", oversized)])

    assert len(knowledge.documents[0].content) == MAX_DOCUMENT_CHARS
    assert len(knowledge.chunks) <= MAX_TOTAL_CHUNKS
