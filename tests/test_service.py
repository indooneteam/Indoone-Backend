import pytest

from app.ai import service


@pytest.mark.asyncio
async def test_service_routes_unseen_requests_to_universal_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            calls.append(user_prompt)
            return "A newly generated answer for an unseen request."

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_knowledge_base", None)

    reply = await service.generate_reply("Explain a topic that is not in any example.")

    assert reply == "A newly generated answer for an unseen request."
    assert len(calls) == 1
    assert "Explain a topic that is not in any example." in calls[0]


@pytest.mark.asyncio
async def test_service_passes_history_and_document_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured.append(user_prompt)
            return "Context was used."

    monkeypatch.setattr(service, "_universal_answer_provider", FakeProvider())
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_knowledge_base", None)

    reply = await service.generate_reply(
        "Summarize the discussion.",
        history=[("user", "We discussed a product launch."), ("assistant", "The launch is next month.")],
        document_context="Release checklist: publish notes.",
    )

    assert reply == "Context was used."
    assert "product launch" in captured[0]
    assert "publish notes" in captured[0]


@pytest.mark.asyncio
async def test_service_fails_safely_without_universal_generation_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_universal_answer_provider", None)
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_knowledge_base", None)

    with pytest.raises(RuntimeError, match="trained Indoone model provider is unavailable"):
        await service.generate_reply("Answer an arbitrary question.")
