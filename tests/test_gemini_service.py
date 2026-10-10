from __future__ import annotations

import asyncio

import pytest

from app.ai import gemini_service


def _response(parts: list[dict[str, object]]) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": parts,
                }
            }
        ]
    }


def test_extract_text_excludes_parts_marked_as_thought() -> None:
    result = gemini_service._extract_text(
        _response(
            [
                {"text": "The user wants a greeting. Plan: greet them.", "thought": True},
                {"text": "Hello! How can I help you?"},
            ]
        )
    )

    assert result == "Hello! How can I help you?"
    assert "Plan:" not in result
    assert "The user wants" not in result


def test_extract_text_rejects_response_containing_only_thoughts() -> None:
    with pytest.raises(RuntimeError, match="empty user-facing answer"):
        gemini_service._extract_text(
            _response([{"text": "Internal plan that must not be shown.", "thought": True}])
        )


def test_gemma4_generation_config_disables_thinking() -> None:
    config = gemini_service._generation_config("gemma-4-26b-a4b-it")

    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert config["maxOutputTokens"] == 512


def test_non_gemma4_generation_config_does_not_set_gemma_thinking_options() -> None:
    config = gemini_service._generation_config("gemini-2.5-flash")

    assert "thinkingConfig" not in config


def test_system_instruction_requires_final_user_facing_answer_only() -> None:
    assert "only the final user-facing answer" in gemini_service._SYSTEM_INSTRUCTION
    assert "Never output hidden analysis" in gemini_service._SYSTEM_INSTRUCTION


def test_generate_gemini_reply_sends_thinking_config_and_returns_only_final(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return _response(
                [
                    {"text": "Internal reasoning. Plan: say hello.", "thought": True},
                    {"text": "Hello! How can I help?"},
                ]
            )

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)

    reply = asyncio.run(gemini_service.generate_gemini_reply("Hi"))

    assert reply == "Hello! How can I help?"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    generation_config = payload["generationConfig"]
    assert isinstance(generation_config, dict)
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert payload["tools"] == [{"googleSearch": {}}]
