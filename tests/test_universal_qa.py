import json

import httpx
import pytest

from app.ai.answer_quality import user_safe_failure
from app.ai.research import ResearchResult
from app.ai.universal_qa import GeminiAnswerProvider, UniversalQuestionAnswerPipeline


def test_gemini_provider_requires_api_key() -> None:
    with pytest.raises(ValueError, match="api_key"):
        GeminiAnswerProvider(" ")


def test_gemini_provider_extracts_all_text_parts() -> None:
    payload = {
        "candidates": [
            {"content": {"parts": [{"text": "First."}, {"text": "Second."}]}}
        ]
    }

    assert GeminiAnswerProvider._extract_text(payload) == "First.\nSecond."


@pytest.mark.asyncio
async def test_gemini_provider_uses_server_side_key_and_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        content = json.dumps(
            {"candidates": [{"content": {"parts": [{"text": "Universal answer."}]}}]}
        ).encode("utf-8")

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return json.loads(self.content)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured["headers"] = kwargs["headers"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    provider = GeminiAnswerProvider("secret", model="gemini-2.5-flash-lite")
    answer = await provider.generate(
        system_instruction="System instruction",
        user_prompt="A brand-new user question.",
    )

    assert answer == "Universal answer."
    assert captured["headers"]["x-goog-api-key"] == "secret"
    assert str(captured["url"]).endswith("/gemini-2.5-flash-lite:generateContent")
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "System instruction"
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "A brand-new user question."


@pytest.mark.asyncio
async def test_universal_pipeline_does_not_need_question_specific_examples() -> None:
    calls: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            calls.append(user_prompt)
            return "Generated from the user's actual request."

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider())

    reply = await pipeline.answer("Describe a concept that has never appeared in the test suite.")

    assert reply == "Generated from the user's actual request."
    assert len(calls) == 1
    assert "never appeared in the test suite" in calls[0]


@pytest.mark.asyncio
async def test_universal_pipeline_collects_live_research_before_answer() -> None:
    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest information about a topic"
            assert limit == 8
            return [
                ResearchResult(
                    "Fresh source",
                    "https://example.com/fresh",
                    "Fresh evidence for the requested topic.",
                ),
                ResearchResult(
                    "Second source",
                    "https://example.org/fresh",
                    "Independent evidence for the requested topic.",
                ),
            ]

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            assert "LIVE RESEARCH REFERENCE:" in user_prompt
            assert "Fresh evidence for the requested topic." in user_prompt
            return "The answer uses the supplied current evidence."

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider(), research_provider=FakeResearch())

    reply = await pipeline.answer("latest information about a topic")

    assert "The answer uses the supplied current evidence." in reply
    assert "Sources:" in reply
    assert "https://example.com/fresh" in reply


@pytest.mark.asyncio
async def test_universal_pipeline_fails_safely_when_current_research_unavailable() -> None:
    class BrokenResearch:
        async def search(self, query: str, limit: int = 5):
            raise RuntimeError("unavailable")

    class FakeProvider:
        async def generate(self, **kwargs):
            raise AssertionError("current answer must not be invented")

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider(), research_provider=BrokenResearch())

    reply = await pipeline.answer("latest information about a topic")

    assert reply == user_safe_failure()
