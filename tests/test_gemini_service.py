from __future__ import annotations

import asyncio

import pytest

from app.ai import gemini_service
from app.ai import tool_loop


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


def test_default_model_is_exact_gemma4_26b_a4b_id(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert gemini_service._get_config() == ("test-key", "gemma-4-26b-a4b-it")


def test_gemma4_generation_config_disables_thinking() -> None:
    config = gemini_service._generation_config("gemma-4-26b-a4b-it")

    assert config["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "maxOutputTokens" not in config


def test_non_gemma4_generation_config_does_not_set_gemma_thinking_options() -> None:
    config = gemini_service._generation_config("gemini-2.5-flash")

    assert "thinkingConfig" not in config


def test_system_instruction_requires_final_user_facing_answer_only() -> None:
    assert "Output only the final answer" in gemini_service._SYSTEM_INSTRUCTION
    assert "no analysis" in gemini_service._SYSTEM_INSTRUCTION


def test_system_instruction_asks_for_plain_text_with_emoji() -> None:
    """The app renders plain text, so markdown markers would show up literally."""
    assert "no markdown" in gemini_service._SYSTEM_INSTRUCTION
    assert "emoji" in gemini_service._SYSTEM_INSTRUCTION
    assert "no markdown" in tool_loop._UNIFIED_SYSTEM_INSTRUCTION
    assert "emoji" in tool_loop._UNIFIED_SYSTEM_INSTRUCTION




def test_build_contents_attaches_media_as_inline_data() -> None:
    import base64

    media_bytes = b"test-image-bytes"
    contents = gemini_service._build_contents(
        "What is in this picture?",
        history=None,
        document_context="",
        media_attachment={"mime_type": "image/jpeg", "content": media_bytes},
    )

    parts = contents[-1]["parts"]
    assert parts[0] == {"text": "What is in this picture?"}
    assert parts[1] == {
        "inlineData": {
            "mimeType": "image/jpeg",
            "data": base64.b64encode(media_bytes).decode("ascii"),
        }
    }


def test_build_contents_attaches_multiple_media_as_inline_data() -> None:
    import base64

    first = {"mime_type": "image/jpeg", "content": b"first"}
    second = {"mime_type": "image/png", "content": b"second"}
    contents = gemini_service._build_contents(
        "Describe both photos",
        history=None,
        document_context="",
        media_attachments=[first, second],
    )

    parts = contents[-1]["parts"]
    assert len(parts) == 3
    assert parts[1]["inlineData"] == {
        "mimeType": "image/jpeg",
        "data": base64.b64encode(b"first").decode("ascii"),
    }
    assert parts[2]["inlineData"] == {
        "mimeType": "image/png",
        "data": base64.b64encode(b"second").decode("ascii"),
    }


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
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemma-4-26b-a4b-it:generateContent"
    )
    assert captured["headers"] == {
        "x-goog-api-key": "test-key",
        "Content-Type": "application/json",
    }
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["systemInstruction"]["parts"][0]["text"] == (
        gemini_service._SYSTEM_INSTRUCTION
        + "\n\n"
        + gemini_service.build_grounded_prompt_instruction()
    )
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "search_web"
    generation_config = payload["generationConfig"]
    assert isinstance(generation_config, dict)
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "minimal"}




def test_generate_gemini_function_call_reply_executes_tool_and_returns_final_answer(monkeypatch) -> None:
    captured_payloads: list[dict[str, object]] = []
    executed: list[tuple[str, dict[str, object]]] = []

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, value: dict[str, object]) -> None:
            self.value = value

        def json(self) -> dict[str, object]:
            return self.value

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            captured_payloads.append(json)
            if len(captured_payloads) == 1:
                return FakeResponse(
                    _response(
                        [
                            {
                                "functionCall": {
                                    "name": "list_recent_emails",
                                    "args": {"provider": "zoho_mail", "limit": 3},
                                    "id": "call-1",
                                }
                            }
                        ]
                    )
                )
            return FakeResponse(_response([{"text": "I found 3 recent emails in Zoho Mail."}]))

    async def execute_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        executed.append((name, arguments))
        return {
            "ok": True,
            "provider": "Zoho Mail",
            "count": 3,
            "emails": [{"subject": "Invoice"}, {"subject": "Finance"}, {"subject": "Meeting"}],
        }

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)

    reply = asyncio.run(
        gemini_service.generate_gemini_function_call_reply(
            "Check my Zoho Mail",
            history=None,
            tool_declarations=[
                {
                    "name": "list_recent_emails",
                    "description": "Get recent emails.",
                    "parameters": {"type": "OBJECT", "properties": {}},
                }
            ],
            execute_tool=execute_tool,
            extra_system_instruction="Use the tool for email requests.",
        )
    )

    assert reply == "I found 3 recent emails in Zoho Mail."
    assert executed == [("list_recent_emails", {"provider": "zoho_mail", "limit": 3})]
    assert captured_payloads[0]["tools"][0]["functionDeclarations"][0]["name"] == "list_recent_emails"
    assert captured_payloads[0]["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    followup_contents = captured_payloads[1]["contents"]
    assert followup_contents[-2]["role"] == "model"
    assert followup_contents[-1]["role"] == "user"
    assert followup_contents[-1]["parts"][0]["functionResponse"]["id"] == "call-1"
    assert followup_contents[-1]["parts"][0]["functionResponse"]["response"]["result"]["count"] == 3



def test_function_call_flow_does_not_trust_unverified_first_answer(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, object]:
            return _response([{"text": "You have 4 new emails."}])

    class FakeAsyncClient:
        def __init__(self, *, timeout: object) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
            return FakeResponse()

    async def unused_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
        raise AssertionError("No tool should be called when Gemini does not request it")

    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("test-key", "gemma-4-26b-a4b-it"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", FakeAsyncClient)

    reply = asyncio.run(
        gemini_service.generate_gemini_function_call_reply(
            "Check my emails",
            history=None,
            tool_declarations=[
                {
                    "name": "list_recent_emails",
                    "description": "Get recent email data.",
                    "parameters": {"type": "OBJECT", "properties": {}},
                }
            ],
            execute_tool=unused_tool,
            extra_system_instruction="Use the email tool.",
        )
    )

    assert "couldn't verify the mailbox data" in reply
    assert "4 new emails" not in reply
