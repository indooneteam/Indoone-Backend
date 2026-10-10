"""Gemini grounding integration tests kept with the web research feature."""

from __future__ import annotations

import asyncio

from app.ai import gemini_service


def test_generate_gemini_reply_returns_grounded_sources(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "A current answer."}]},
                        "groundingMetadata": {
                            "groundingChunks": [
                                {
                                    "web": {
                                        "title": "Official update",
                                        "uri": "https://example.com/update",
                                    }
                                }
                            ]
                        },
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            assert json["tools"] == [{"googleSearch": {}}]
            return FakeResponse()

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)

    reply = asyncio.run(gemini_service.generate_gemini_reply("What is the latest update?"))

    assert reply == (
        "A current answer.\n\nSources:\n"
        "- [Official update](https://example.com/update)"
    )



def test_gemini_search_policy_uses_web_selectively() -> None:
    instruction = gemini_service._SYSTEM_INSTRUCTION
    assert "Do not search the web for stable, evergreen questions" in instruction
    assert "time-sensitive, news, price, availability" in instruction
    assert "Never invent source URLs or citations" in instruction
