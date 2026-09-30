import pytest

from app.ai import service
from app.ai.knowledge import KnowledgeDocument, LocalKnowledgeBase


@pytest.mark.asyncio
async def test_service_supplies_local_knowledge_as_reference_not_as_hardcoded_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured.append(user_prompt)
            return "The answer was generated from the reference context."

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(
        service,
        "_knowledge_base",
        LocalKnowledgeBase(
            [KnowledgeDocument("doc", "Indoone architecture", "Reference fact about the product architecture.")]
        ),
    )

    reply = await service.generate_reply("Explain the product architecture.")

    assert reply == "The answer was generated from the reference context."
    assert "Reference fact about the product architecture." in captured[0]
    assert "LOCAL KNOWLEDGE REFERENCE:" in captured[0]
