import pytest

from app.ai import service
from app.ai.research import ResearchResult


@pytest.mark.asyncio
async def test_service_adds_sources_to_a_universal_current_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured.append(user_prompt)
            assert "LIVE RESEARCH REFERENCE:" in user_prompt
            return "The answer is grounded in the supplied current evidence."

    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest Indoone news"
            assert limit == 8
            return [
                ResearchResult("Source one", "https://example.com/one", "Fresh evidence one."),
                ResearchResult("Source two", "https://example.org/two", "Fresh evidence two."),
            ]

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", FakeResearch())
    monkeypatch.setattr(service, "_knowledge_base", None)

    reply = await service.generate_reply("latest Indoone news")

    assert reply.startswith("The answer is grounded")
    assert "Sources:" in reply
    assert "https://example.com/one" in reply
    assert len(captured) == 1
