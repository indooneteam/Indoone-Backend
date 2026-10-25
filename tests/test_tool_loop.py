import json

import pytest

from app.ai.execution import tool_loop
from app.ai.execution.approval import issue_approval_token
from app.ai.execution.tools import ToolResult


@pytest.mark.asyncio
async def test_dispatch_routes_github_tools(monkeypatch) -> None:
    seen = {}

    async def fake(user_id, name, arguments):
        seen["call"] = (user_id, name, arguments)
        return {"ok": True, "result": {"repositories": []}}

    monkeypatch.setattr("app.plugins.github.chat_tools._execute_github_tool", fake)

    result = await tool_loop.dispatch_tool("u1", "c1", "github_list_repositories", {"per_page": 5})
    assert result["ok"] is True
    assert seen["call"][1] == "github_list_repositories"


@pytest.mark.asyncio
async def test_dispatch_routes_email_tools(monkeypatch) -> None:
    seen = {}

    async def fake(user_id, conversation_id, name, arguments, *, original_message=""):
        seen["call"] = (user_id, conversation_id, name, original_message)
        return {"ok": True, "result": {}}

    monkeypatch.setattr("app.plugins.email_tools.execute_email_tool", fake)

    result = await tool_loop.dispatch_tool(
        "u1", "c1", "search_emails", {"query": "invoice"}, original_message="find my invoice mail"
    )
    assert result["ok"] is True
    assert seen["call"][2] == "search_emails"
    assert seen["call"][3] == "find my invoice mail"


@pytest.mark.asyncio
async def test_dispatch_routes_core_text_tools(monkeypatch) -> None:
    seen = {}

    async def fake_run_tool_async(name, payload, **kwargs):
        seen["call"] = (name, payload)
        return ToolResult(name=name, output="42", safe=True)

    monkeypatch.setattr(tool_loop, "run_tool_async", fake_run_tool_async)

    result = await tool_loop.dispatch_tool("u1", "c1", "calculator", {"expression": "6*7"})
    assert result == {"ok": True, "result": 42}
    assert seen["call"] == ("calculator", "6*7")


@pytest.mark.asyncio
async def test_dispatch_routes_search_web(monkeypatch) -> None:
    from app.web_research.research import ResearchResult
    from app.web_research.tools import SearchWebToolResult

    async def fake_search(call):
        assert call.query == "latest Indoone news"
        return SearchWebToolResult(
            status="ok", results=(ResearchResult(title="T", url="http://x", snippet="s"),)
        )

    monkeypatch.setattr("app.web_research.tools.execute_search_web_call", fake_search)

    result = await tool_loop.dispatch_tool("u1", "c1", "search_web", {"query": "latest Indoone news"})
    assert result["ok"] is True
    assert result["result"]["results"][0]["title"] == "T"


@pytest.mark.asyncio
async def test_dispatch_reports_unavailable_search(monkeypatch) -> None:
    from app.web_research.tools import SearchWebToolResult

    async def fake_search(call):
        return SearchWebToolResult(status="unavailable")

    monkeypatch.setattr("app.web_research.tools.execute_search_web_call", fake_search)

    result = await tool_loop.dispatch_tool("u1", "c1", "search_web", {"query": "x"})
    assert result["ok"] is False
    assert result["code"] == "SEARCH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_dispatch_blocks_write_tools_without_approval() -> None:
    result = await tool_loop.dispatch_tool("u1", "c1", "sandbox_execution", {"code": "print(1)"})
    assert result["ok"] is False
    assert result["code"] == "APPROVAL_REQUIRED"


@pytest.mark.asyncio
async def test_dispatch_runs_write_tool_with_valid_approval(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "s" * 40)
    token = issue_approval_token("u1", "sandbox_execution")

    async def fake_run_tool_async(name, payload, **kwargs):
        return ToolResult(name=name, output=json.dumps({"ran": True}), safe=True)

    monkeypatch.setattr(tool_loop, "run_tool_async", fake_run_tool_async)

    result = await tool_loop.dispatch_tool(
        "u1", "c1", "sandbox_execution", {"code": "print(1)"}, approval_tokens=[token]
    )
    assert result == {"ok": True, "result": {"ran": True}}


@pytest.mark.asyncio
async def test_write_tool_token_is_passed_into_the_payload(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "s" * 40)
    # Pretend the GitHub connector is connected and authorized for the write.
    monkeypatch.setattr(
        "app.ai.execution.approval.authorize_tool",
        lambda *args, **kwargs: SimpleNamespace(allowed=True, reason=""),
    )
    token = issue_approval_token("u1", "github_update_repository_file")
    seen = {}

    async def fake_run_tool_async(name, payload, **kwargs):
        seen["payload"] = json.loads(payload)
        return ToolResult(name=name, output="{}", safe=True)

    monkeypatch.setattr(tool_loop, "run_tool_async", fake_run_tool_async)

    result = await tool_loop.dispatch_tool(
        "u1",
        "c1",
        "github_update_repository_file",
        {"repository": "o/r", "path": "a.py", "content": "x", "message": "m"},
        approval_tokens=[token],
    )
    assert result["ok"] is True
    assert seen["payload"]["approval_token"] == token


@pytest.mark.asyncio
async def test_dispatch_rejects_unknown_tools() -> None:
    unknown = await tool_loop.dispatch_tool("u1", "c1", "does_not_exist", {})
    assert unknown["ok"] is False


@pytest.mark.asyncio
async def test_run_tool_loop_is_a_no_op_without_gemini(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_MODEL_BACKEND", raising=False)
    assert await tool_loop.run_tool_loop("u1", "c1", "hello") is None


@pytest.mark.asyncio
async def test_run_tool_loop_uses_auto_mode_and_the_users_declarations(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "gemini")
    captured = {}

    async def fake_gemini(
        message,
        *,
        history,
        tool_declarations,
        execute_tool,
        extra_system_instruction,
        tool_mode="ANY",
        return_none_when_no_tool=False,
        max_tool_rounds=4,
        reuse_first_text_when_no_tool=False,
    ):
        captured["mode"] = tool_mode
        captured["count"] = len(tool_declarations)
        captured["return_none"] = return_none_when_no_tool
        captured["reuse_text"] = reuse_first_text_when_no_tool
        return "done"

    monkeypatch.setattr(
        "app.ai.gateway.gemini_service.generate_gemini_function_call_reply", fake_gemini
    )

    out = await tool_loop.run_tool_loop("u1", "c1", "list my repositories", history=[])
    assert out == "done"
    assert captured["mode"] == "AUTO"
    assert captured["return_none"] is True
    assert captured["reuse_text"] is True
    assert captured["count"] == len(tool_loop.model_tool_declarations("u1"))


def test_plain_chat_does_not_carry_the_tool_catalogue() -> None:
    """A bare greeting must not pay ~2,900 tokens for tools it cannot use.

    "how are you" used to be in this list. It is not any more, and that is
    deliberate: a three-word question like "how are you" and a three-word
    question like "nale enu special" are the same length, and only the model can
    tell them apart. We send the catalogue and let it choose, rather than guess
    from the wording and get it wrong.
    """
    assert tool_loop.needs_tools("hi") is False
    assert tool_loop.needs_tools("thanks!") is False
    assert tool_loop.needs_tools("ok") is False


def test_a_request_for_a_service_still_uses_tools() -> None:
    assert tool_loop.needs_tools("check my Zoho Mail") is True
    assert tool_loop.needs_tools("how many GitHub repositories do I have?") is True
    assert tool_loop.needs_tools("summarise my unread email") is True
    assert tool_loop.needs_tools("ಇಮೇಲ್ ನೋಡು") is True
    assert tool_loop.needs_tools("what is the capital of Karnataka and why") is True


def test_tool_declarations_skip_connectors_the_user_has_not_connected(monkeypatch) -> None:
    """A connector the user never connected can only fail, so it is not offered."""
    monkeypatch.setattr(
        tool_loop,
        "get_integration_token_metadata",
        lambda user_id, integration_id: {} if integration_id == "github" else None,
    )

    names = [item["name"] for item in tool_loop.model_tool_declarations("u1")]
    assert "github_list_repositories" in names
    assert "canva_list_designs" not in names
    # Without a user context the whole catalogue is returned.
    assert len(tool_loop.model_tool_declarations()) > len(names)


def test_a_festival_question_gets_the_tools():
    """A festival's date is year-specific, so it is a current-information question.

    Before this, a festival question carried no tools, the model had no way to look
    the date up, and it answered that it did not know.
    """
    from app.ai.execution.tool_loop import needs_tools

    for question in (
        "ivattu yava habba ide",
        "when is diwali this year",
        "deepavali date 2026",
        "ಇವತ್ತು ಯಾವ ಹಬ್ಬ ಇದೆ",
        "ಶುಭ ಮುಹೂರ್ತ ಯಾವಾಗ",
    ):
        assert needs_tools(question) is True, question

def test_the_gate_does_not_guess_from_topic_words():
    """The gate is about length, not about knowing every festival word.

    A word list would need twelve languages and eleven scripts, and every word
    added for one language is a gap in the next. Any real sentence carries the
    catalogue and the model chooses; only a bare greeting is spared the cost.
    """
    from app.ai.execution.tool_loop import _TOOL_HINTS, needs_tools

    for word in ("festival", "habba", "deepavali", "muhurta", "ಹಬ್ಬ", "ದೀಪಾವಳಿ"):
        assert word not in _TOOL_HINTS, word

    # and yet every one of these still gets the catalogue
    for question in ("ivattu yava habba ide", "nale enu special", "when is diwali"):
        assert needs_tools(question) is True, question

    # a bare greeting is still spared
    for greeting in ("hi", "ok", "thanks"):
        assert needs_tools(greeting) is False, greeting
