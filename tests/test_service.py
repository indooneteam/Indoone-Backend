import pytest

from app.ai import service

@pytest.fixture(autouse=True)
def disable_live_web_research(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_WEB_RESEARCH_ENABLED", False)


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


@pytest.mark.asyncio
async def test_live_web_research_is_added_before_model_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_WEB_RESEARCH_ENABLED", True)
    captured: dict[str, str] = {}

    class FakeResearchResult:
        title = "Fresh web result"
        url = "https://example.com/fresh"
        snippet = "Current evidence from the live web."

    class FakeResearchProvider:
        async def search(self, query: str, limit: int = 5):
            assert query == "latest AI news"
            assert limit == 5
            return [FakeResearchResult()]

    monkeypatch.setattr(service, "build_research_provider", lambda: FakeResearchProvider())

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured["prompt"] = user_prompt
            return "Grounded answer."

    monkeypatch.setattr(service, "_model_answer_provider", FakeProvider())
    monkeypatch.delenv("INDOONE_MODEL_BACKEND", raising=False)

    reply = await service.generate_reply("latest AI news")

    assert reply == "Grounded answer."
    assert "LIVE WEB RESEARCH EVIDENCE:" in captured["prompt"]
    assert "example.com" in captured["prompt"]
    assert "Current evidence from the live web." in captured["prompt"]


@pytest.mark.asyncio
async def test_live_web_research_failure_falls_back_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResearchProvider:
        async def search(self, query: str, limit: int = 5):
            raise RuntimeError("network unavailable")

    monkeypatch.setattr(service, "build_research_provider", lambda: FakeResearchProvider())
    captured: dict[str, str] = {}

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            captured["prompt"] = user_prompt
            return "Model-only answer."

    monkeypatch.setattr(service, "_model_answer_provider", FakeProvider())
    monkeypatch.delenv("INDOONE_MODEL_BACKEND", raising=False)

    reply = await service.generate_reply("hello")

    assert reply == "Model-only answer."
    assert "LIVE WEB RESEARCH EVIDENCE:" not in captured["prompt"]


def test_model_artifacts_require_indoone_model_release(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(service, "get_github_release_storage", lambda: None)

    with pytest.raises(
        RuntimeError,
        match="Indoone model serving requires GITHUB_MODEL_REPOSITORY, GITHUB_MODEL_RELEASE_TAG, and GITHUB_MODEL_TOKEN",
    ):
        service._ensure_model_artifacts(tmp_path)


@pytest.mark.asyncio
async def test_service_routes_to_qwen_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "qwen")

    async def fake_generate_qwen_reply(message, history=None, document_context=""):
        assert message == "Hello Qwen"
        return "Qwen response"

    from app.ai import qwen_service
    monkeypatch.setattr(qwen_service, "generate_qwen_reply", fake_generate_qwen_reply)

    reply = await service.generate_reply("Hello Qwen")
    assert reply == "Qwen response"
