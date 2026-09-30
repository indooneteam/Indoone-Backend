import pytest

from app.ai import service


@pytest.mark.asyncio
async def test_service_sends_the_actual_request_directly_to_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured["system_instruction"] = system_instruction
            captured["user_prompt"] = user_prompt
            captured["temperature"] = temperature
            captured["max_output_tokens"] = max_output_tokens
            return "A model-generated answer."

    monkeypatch.setattr(service, "_model_answer_provider", FakeProvider())

    reply = await service.generate_reply("Explain a topic that is not in any example.")

    assert reply == "A model-generated answer."
    assert captured["user_prompt"] == "Explain a topic that is not in any example."
    assert captured["system_instruction"] == ""


@pytest.mark.asyncio
async def test_service_only_adds_explicit_conversation_and_document_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured.append(user_prompt)
            return "Context was used."

    monkeypatch.setattr(service, "_model_answer_provider", FakeProvider())

    reply = await service.generate_reply(
        "Summarize the discussion.",
        history=[("user", "We discussed a product launch."), ("assistant", "The launch is next month.")],
        document_context="Release checklist: publish notes.",
    )

    assert reply == "Context was used."
    assert "product launch" in captured[0]
    assert "publish notes" in captured[0]


@pytest.mark.asyncio
async def test_local_model_provider_has_no_generation_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntime:
        def generate(self, *args, **kwargs) -> str:
            return "slow-model-answer"

    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())
    provider = service._LocalModelAnswerProvider()

    reply = await provider.generate(
        system_instruction="You are Indoone AI.",
        user_prompt="Take as long as needed.",
    )

    assert reply == "slow-model-answer"
    assert not hasattr(service, "_MODEL_GENERATION_TIMEOUT_SECONDS")


def test_model_artifacts_require_indoone_model_release(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(service, "get_github_release_storage", lambda: None)

    with pytest.raises(
        RuntimeError,
        match="Indoone model serving requires GITHUB_MODEL_REPOSITORY, GITHUB_MODEL_RELEASE_TAG, and GITHUB_MODEL_TOKEN",
    ):
        service._ensure_model_artifacts(tmp_path)
