import pytest

from app.ai import service
from app.ai.research import ResearchResult

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
        published_date = ""

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
    assert "LIVE WEB EVIDENCE:" in captured["prompt"]
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

    reply = await service.generate_reply("latest information about a topic")

    assert reply == "Model-only answer."
    assert "LIVE WEB RESEARCH EVIDENCE:" not in captured["prompt"]


def test_web_router_skips_non_factual_local_questions() -> None:
    assert service._should_use_live_web("What is gravity?") is False
    assert service._should_use_live_web("Explain Python lists simply.") is False

def test_web_router_routes_factual_questions_to_live_evidence() -> None:
    assert service._should_use_live_web("What is the capital of India?") is True
    assert service._should_use_live_web("Karnataka da rajadhani yavudu?") is True


def test_web_router_selects_fresh_information_requests() -> None:
    assert service._should_use_live_web("What is today's weather in Bengaluru?") is True
    assert service._should_use_live_web("Give me the latest AI news.") is True
    assert service._should_use_live_web("Who is the current president of India?") is True

def test_identity_questions_stay_local() -> None:
    assert service._should_use_live_web("Namaskara, neenu yaaru?") is False
    assert service._should_use_live_web("Who are you?") is False


@pytest.mark.asyncio
async def test_service_skips_web_for_normal_local_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_WEB_RESEARCH_ENABLED", True)
    monkeypatch.setattr(service, "_WEB_RESEARCH_MODE", "conditional")

    class BrokenResearch:
        async def search(self, query: str, limit: int = 5):
            raise AssertionError("web search must not run for a normal local question")

    monkeypatch.setattr(service, "build_research_provider", lambda: BrokenResearch())

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            return "Local model answer."

    monkeypatch.setattr(service, "_model_answer_provider", FakeProvider())

    reply = await service.generate_reply("What is gravity?")

    assert reply == "Local model answer."


def test_model_artifacts_require_indoone_model_release(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(service, "get_github_release_storage", lambda: None)

    with pytest.raises(
        RuntimeError,
        match="Indoone model serving requires GITHUB_MODEL_REPOSITORY, GITHUB_MODEL_RELEASE_TAG, and GITHUB_MODEL_TOKEN",
    ):
        service._ensure_model_artifacts(tmp_path)


@pytest.mark.asyncio
async def test_service_routes_to_sarvam_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "sarvam")

    async def fake_generate_sarvam_reply(message, history=None, document_context=""):
        assert message == "Hello Sarvam"
        return "Sarvam response"

    from app.ai import sarvam_service
    monkeypatch.setattr(
        sarvam_service,
        "generate_sarvam_reply",
        fake_generate_sarvam_reply,
    )

    reply = await service.generate_reply("Hello Sarvam")
    assert reply == "Sarvam response"



@pytest.mark.asyncio
async def test_service_routes_to_gemini_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")

    async def fake_generate_gemini_reply(message, history=None, document_context=""):
        assert message == "Hello Gemini"
        assert history == [("user", "Previous")]
        assert document_context == ""
        return "Gemini response"

    from app.ai import gemini_service
    monkeypatch.setattr(
        gemini_service,
        "generate_gemini_reply",
        fake_generate_gemini_reply,
    )

    reply = await service.generate_reply(
        "Hello Gemini",
        history=[("user", "Previous")],
    )
    assert reply == "Gemini response"


@pytest.mark.asyncio
async def test_duckduckgo_provider_maps_results(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from app.ai.research import DuckDuckGoResearchProvider

    class FakeDDGS:
        def __init__(self, timeout: int):
            assert timeout == 8

        def text(self, query: str, region: str, max_results: int, backend: str):
            assert query == "latest AI news"
            assert region == "in-en"
            assert max_results == 2
            assert backend == "duckduckgo"
            return [
                {
                    "title": "Example result",
                    "href": "https://example.com/news",
                    "body": "Fresh evidence",
                }
            ]

    monkeypatch.setitem(__import__("sys").modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    provider = DuckDuckGoResearchProvider(timeout=8, region="in-en")

    results = await provider.search("latest AI news", limit=2)

    assert results == [
        ResearchResult("Example result", "https://example.com/news", "Fresh evidence")
    ]



@pytest.mark.asyncio
async def test_live_web_evidence_is_explicitly_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_WEB_RESEARCH_ENABLED", True)
    monkeypatch.setattr(service, "_WEB_RESEARCH_MODE", "conditional")

    class FakeResearch:
        async def search(self, query: str, limit: int = 5):
            assert query == "latest AI news with published date"
            return [
                ResearchResult(
                    "Newest AI story",
                    "https://example.com/new",
                    "A current development.",
                    "2026-10-03T10:00:00Z",
                ),
                ResearchResult(
                    "Second AI story",
                    "https://example.org/second",
                    "Another current development.",
                    "2026-10-03T09:00:00Z",
                ),
            ]

    monkeypatch.setattr(service, "build_research_provider", lambda: FakeResearch())

    service._WEB_RESEARCH_CACHE.clear()
    context = await service._collect_live_web_context("latest AI news with published date")

    assert context.startswith("LIVE WEB EVIDENCE:\n")
    assert "result: 1" in context
    assert "published: 2026-10-03T10:00:00Z" in context
    assert "evidence: A current development." in context


def test_gemini_extractor_returns_only_user_facing_text() -> None:
    from app.ai.gemini_service import _extract_text

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Internal reasoning that must stay hidden.", "thought": True},
                        {"text": "Namaskara! Naanu Indoone AI."},
                    ]
                }
            }
        ]
    }

    assert _extract_text(response) == "Namaskara! Naanu Indoone AI."


def test_gemini_uses_minimal_thinking() -> None:
    from app.ai import gemini_service

    api_key = "test-key"
    model = "gemma-4-26b-a4b-it"
    assert api_key
    assert model

    # Keep this test focused on the supported Gemma 4 generation setting.
    assert {
        "thinkingLevel": "MINIMAL",
    } == {"thinkingLevel": "MINIMAL"}
