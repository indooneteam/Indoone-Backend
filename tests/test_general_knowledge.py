from pathlib import Path

import pytest
import httpx

from app.ai.general_knowledge import (
    DEFAULT_USER_AGENT,
    WikipediaAnswer,
    WikipediaKnowledgeProvider,
    _question_to_topic,
    is_general_knowledge_question,
)


@pytest.mark.parametrize(
    "question",
    [
        "What is photosynthesis?",
        "Who is Ada Lovelace?",
        "Tell me about Alan Turing.",
        "Explain gravity in simple words.",
        "How many states are in India?",
        "How much water is on Earth?",
        "What causes day and night?",
        "What is the difference between mass and weight?",
        "ಭೂಮಿ ಏಕೆ ತಿರುಗುತ್ತದೆ?",
    ],
)
def test_general_knowledge_questions_are_detected(question: str) -> None:
    assert is_general_knowledge_question(question)


@pytest.mark.parametrize(
    "question",
    [
        "Write a Python function.",
        "Give me a project plan.",
        "What is the latest news?",
        "Who is the president of India?",
        "Calculate 12 + 7.",
    ],
)
def test_non_factual_workflows_are_not_routed_to_general_knowledge(question: str) -> None:
    assert not is_general_knowledge_question(question)


@pytest.mark.asyncio
async def test_wikipedia_provider_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload, status_code: int = 200, url: str = "https://example.test"):
            self._payload = payload
            self.status_code = status_code
            self.request = httpx.Request("GET", url)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, **kwargs):
            self.headers = kwargs["headers"]
            self.summary_fallback = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            if "/w/api.php" in url:
                assert self.headers["User-Agent"] == DEFAULT_USER_AGENT
                assert self.headers["Api-User-Agent"] == DEFAULT_USER_AGENT
                return FakeResponse(
                    {"query": {"search": [{"title": "Photosynthesis"}]}}
                )
            return FakeResponse(
                {
                    "extract": "Photosynthesis is the process by which plants use light energy to make food.",
                    "content_urls": {
                        "desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}
                    },
                },
                url=url,
            )

    monkeypatch.setattr(
        "app.ai.general_knowledge.httpx.AsyncClient",
        lambda **kwargs: FakeClient(**kwargs),
    )

    result = await WikipediaKnowledgeProvider().answer("What is photosynthesis?")

    assert result == WikipediaAnswer(
        title="Photosynthesis",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        extract="Photosynthesis is the process by which plants use light energy to make food.",
    )


def test_question_to_topic_removes_common_english_prefixes() -> None:
    assert _question_to_topic("What is photosynthesis?") == "photosynthesis"
    assert _question_to_topic("Who was Alan Turing?") == "Alan Turing"
    assert _question_to_topic("Explain gravity in simple words.") == "gravity"
    assert _question_to_topic("How many states are in India?") == "states are in India"
    assert _question_to_topic("How much water is on Earth?") == "water is on Earth"


@pytest.mark.asyncio
async def test_wikipedia_provider_uses_direct_summary_when_search_is_forbidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __init__(self, payload, status_code: int = 200, url: str = "https://example.test"):
            self._payload = payload
            self.status_code = status_code
            self.request = httpx.Request("GET", url)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {self.status_code}",
                    request=self.request,
                    response=httpx.Response(self.status_code, request=self.request),
                )

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, **kwargs):
            self.headers = kwargs["headers"]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            if "/w/api.php" in url:
                return FakeResponse({}, status_code=403, url=url)
            assert "/api/rest_v1/page/summary/photosynthesis" in url
            return FakeResponse(
                {
                    "extract": "Photosynthesis is the process by which green plants convert light energy into chemical energy.",
                    "content_urls": {
                        "desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}
                    },
                },
                url=url,
            )

    monkeypatch.setattr(
        "app.ai.general_knowledge.httpx.AsyncClient",
        lambda **kwargs: FakeClient(**kwargs),
    )

    result = await WikipediaKnowledgeProvider().answer("What is photosynthesis?")

    assert result == WikipediaAnswer(
        title="photosynthesis",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        extract="Photosynthesis is the process by which green plants convert light energy into chemical energy.",
    )