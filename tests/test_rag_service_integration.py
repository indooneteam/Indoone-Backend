import pytest

from app.ai import service
from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase


@pytest.mark.asyncio
async def test_rag_context_is_supplied_to_universal_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    knowledge = LocalKnowledgeBase(
        [KnowledgeDocument("pricing", "Pricing", "Reference price information.")]
    )
    captured: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured.append(user_prompt)
            return "A generated support reply based on the supplied context."

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_knowledge_base", knowledge)

    result = await service.LocalAIService().generate(
        "Write a short support reply about the product price."
    )

    assert result == "A generated support reply based on the supplied context."
    assert "LOCAL KNOWLEDGE REFERENCE:" in captured[0]
    assert "Reference price information." in captured[0]
