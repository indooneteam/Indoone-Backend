import pytest

from app.ai import service
from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase


@pytest.mark.asyncio
async def test_rag_context_is_injected_into_generation(monkeypatch) -> None:
    knowledge = LocalKnowledgeBase([
        KnowledgeDocument("pricing", "Pricing", "Indoone Pro costs 499 rupees per month.")
    ])
    captured: dict[str, str] = {}

    async def fake_generate(context: str) -> str:
        captured["context"] = context
        return "The documented price is 499 rupees per month."

    monkeypatch.setattr(service, "_knowledge_base", knowledge)
    monkeypatch.setattr(service, "_runtime", None)
    monkeypatch.setattr(service._fallback_engine, "generate", fake_generate)

    result = await service.LocalAIService().generate("What is the Indoone Pro price?")

    assert result == "The documented price is 499 rupees per month."
    assert "Relevant knowledge:" in captured["context"]
    assert "[pricing:1]" in captured["context"]
    assert "499 rupees per month" in captured["context"]
