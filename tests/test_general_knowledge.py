from pathlib import Path

import pytest

from app.ai.general_knowledge import (
    WikipediaAnswer,
    WikipediaKnowledgeProvider,
    is_general_knowledge_question,
)


@pytest.mark.parametrize(
    "question",
    [
        "What is photosynthesis?",
        "Who is Ada Lovelace?",
        "Explain gravity in simple words.",
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
        "Calculate 12 + 7.",
    ],
)
def test_non_factual_workflows_are_not_routed_to_general_knowledge(question: str) -> None:
    assert not is_general_knowledge_question(question)


@pytest.mark.asyncio
async def test_wikipedia_provider_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, **kwargs):
            if "/w/api.php" in url:
                return FakeResponse(
                    {"query": {"search": [{"title": "Photosynthesis"}]}}
                )
            return FakeResponse(
                {
                    "extract": "Photosynthesis is the process by which plants use light energy to make food.",
                    "content_urls": {
                        "desktop": {"page": "https://en.wikipedia.org/wiki/Photosynthesis"}
                    },
                }
            )

    monkeypatch.setattr("app.ai.general_knowledge.httpx.AsyncClient", lambda **kwargs: FakeClient())

    result = await WikipediaKnowledgeProvider().answer("What is photosynthesis?")

    assert result == WikipediaAnswer(
        title="Photosynthesis",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        extract="Photosynthesis is the process by which plants use light energy to make food.",
    )
