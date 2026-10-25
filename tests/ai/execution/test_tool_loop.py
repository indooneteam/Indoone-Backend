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

    monkeypatch.setattr("app.plugins.email.email_tools.execute_email_tool", fake)

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
    token = issue_approval_token("u1", "github_delete_branch")
    seen = {}

    async def fake_run_tool_async(name, payload, **kwargs):
        seen["payload"] = json.loads(payload)
        return ToolResult(name=name, output="{}", safe=True)

    monkeypatch.setattr(tool_loop, "run_tool_async", fake_run_tool_async)

    result = await tool_loop.dispatch_tool(
        "u1",
        "c1",
        "github_delete_branch",
        {"repository": "o/r", "branch": "x", "message": "m"},
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
        # A callable: the loop asks each round so discovery can add to it.
        captured["count"] = len(tool_declarations())
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


def test_the_whole_catalogue_is_not_sent_any_more() -> None:
    """Four small declarations, not forty.

    Every declaration is re-sent on every round of every message. The whole
    catalogue measured 41 declarations and about 4,863 tokens; this is the number
    that keeps it from creeping back.
    """
    import json

    declarations = tool_loop.model_tool_declarations("u1")
    names = [item["name"] for item in declarations]

    assert names == ["find_tools", "calculator", "text_stats", "search_web"]
    assert len(json.dumps(declarations)) // 4 < 600


def test_a_discovered_tool_is_offered_on_the_next_round() -> None:
    names = [item["name"] for item in tool_loop.model_tool_declarations("u1", discovered=("read_email",))]

    assert "read_email" in names
    assert "find_tools" in names
    assert "github_list_repositories" not in names


def test_find_tools_finds_the_github_tools() -> None:
    result = tool_loop._find_tools_result("list my github repositories", "u1")

    assert result["ok"] is True
    names = [item["name"] for item in result["result"]["tools"]]
    assert "github_list_repositories" in names


def test_discovery_is_not_hidden_by_a_missing_connector() -> None:
    """The tool is offered even when the connector is not connected.

    Hiding it made the model answer "I don't have access" for something the product
    does, and it had no way to tell the user how to fix it. The executor answers with
    "connect it in Plugins" instead, which is a reply the user can act on.
    """
    result = tool_loop._find_tools_result("list my github repositories", "a-user-with-nothing-connected")

    assert result["ok"] is True
    names = [item["name"] for item in result["result"]["tools"]]
    assert "github_list_repositories" in names


def test_find_tools_says_nothing_matches_a_story() -> None:
    """A writing request needs no tool, and the search says so instead of guessing."""
    result = tool_loop._find_tools_result("write a story about a king", "")

    assert result["ok"] is True
    assert result["result"]["tools"] == []


def test_find_tools_needs_a_query() -> None:
    result = tool_loop._find_tools_result("   ", "u1")

    assert result["ok"] is False
    assert result["code"] == "EMPTY_QUERY"


def test_the_tool_instruction_carries_the_answer_rules() -> None:
    """Every message goes through the tool path now, so it needs the answer rules.

    A festival date or the persona must not depend on which path a message took.
    """
    from app.ai.gateway.ai_rules import ANSWER_SYSTEM_INSTRUCTION, TOOL_SYSTEM_INSTRUCTION

    assert TOOL_SYSTEM_INSTRUCTION.startswith(ANSWER_SYSTEM_INSTRUCTION)
    assert "find_tools" in TOOL_SYSTEM_INSTRUCTION
