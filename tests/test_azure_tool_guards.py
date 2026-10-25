"""The Azure loop must apply the same guards as the Gemini loop.

It used to run an undeclared tool and to check the call cap only after executing a
whole round, so a model on Azure could issue more calls than intended and a name it
was never offered still reached the backend.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.gateway import azure_openai_service as az
_DECLARATIONS = [
    {
        "name": "calculator",
        "description": "Evaluate arithmetic.",
        "parameters": {"type": "OBJECT", "properties": {"expression": {"type": "STRING"}}},
    }
]


def _tool_call(name: str, call_id: str = "1") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}


def _text_response(text: str) -> dict:
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


def _call_response(*calls: dict) -> dict:
    return {"choices": [{"message": {"content": None, "tool_calls": list(calls)}}]}


@pytest.fixture
def _azure(monkeypatch):
    monkeypatch.setattr(az, "_get_tool_config", lambda: ("key", "https://example", "deployment", "2024-01-01"))
    sent: list[dict] = []

    def install(responses: list[dict]):
        async def fake_post(url, api_key, payload):
            sent.append(payload)
            return responses.pop(0) if responses else _text_response("done")

        monkeypatch.setattr(az, "_post_chat", fake_post)
        return sent

    return install


def test_azure_refuses_an_undeclared_tool(_azure):
    sent = _azure([_call_response(_tool_call("not_a_tool")), _text_response("done")])
    ran: list[str] = []

    async def execute_tool(name, arguments):
        ran.append(name)
        return {"ok": True}

    asyncio.run(az.generate_azure_function_call_reply(
        "hi",
        history=None,
        tool_declarations=_DECLARATIONS,
        execute_tool=execute_tool,
        extra_system_instruction="",
        tool_mode="AUTO",
        return_none_when_no_tool=False,
    ))

    assert ran == [], "an undeclared tool must never run"
    tool_messages = [m for m in sent[1]["messages"] if m.get("role") == "tool"]
    assert tool_messages, "the loop must answer the call with a tool result"
    assert json.loads(tool_messages[0]["content"])["code"] == "UNKNOWN_TOOL"


def test_azure_stops_at_the_call_cap(_azure):
    # Nine calls to a declared tool in one round: eight may run, the ninth is
    # refused - the same cap the Gemini loop enforces before executing.
    calls = [_tool_call("calculator", f"c{index}") for index in range(9)]
    _azure([_call_response(*calls)])
    ran: list[str] = []

    async def execute_tool(name, arguments):
        ran.append(name)
        return {"ok": True}

    asyncio.run(az.generate_azure_function_call_reply(
        "hi",
        history=None,
        tool_declarations=_DECLARATIONS,
        execute_tool=execute_tool,
        extra_system_instruction="",
        tool_mode="AUTO",
        return_none_when_no_tool=False,
    ))

    assert len(ran) == 8, "only the capped number of tool calls may run"
