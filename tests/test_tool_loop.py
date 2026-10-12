import pytest

from app.ai import tool_loop
from app.ai.tools import ToolResult


def test_declarations_are_one_list_without_the_legacy_gmail_family() -> None:
    declarations = tool_loop.model_tool_declarations()
    names = [item["name"] for item in declarations]

    # The legacy gmail_* family is not exposed; the email plugin supersedes it.
    assert "gmail_send" not in names
    assert "gmail_compose" not in names

    # GitHub read tools, email plugin tools and local tools are all present.
    assert "github_list_repositories" in names
    assert "github_read_repository_file" in names
    assert "search_emails" in names
    assert "prepare_email_to_send" in names
    assert "calculator" in names

    # Approval-required tools are not handed to the model yet.
    assert "sandbox_execution" not in names
    assert "github_update_repository_file" not in names

    # No duplicates, and every entry is fully described.
    assert len(names) == len(set(names))
    for item in declarations:
        assert item["description"]
        assert item["parameters"]


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
async def test_dispatch_rejects_unknown_and_approval_tools() -> None:
    approval = await tool_loop.dispatch_tool("u1", "c1", "sandbox_execution", {"code": "x"})
    assert approval["ok"] is False
    unknown = await tool_loop.dispatch_tool("u1", "c1", "does_not_exist", {})
    assert unknown["ok"] is False


@pytest.mark.asyncio
async def test_run_tool_loop_is_a_no_op_without_gemini(monkeypatch) -> None:
    monkeypatch.delenv("INDOONE_MODEL_BACKEND", raising=False)
    assert await tool_loop.run_tool_loop("u1", "c1", "hello") is None


@pytest.mark.asyncio
async def test_run_tool_loop_uses_auto_mode_and_all_declarations(monkeypatch) -> None:
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
    ):
        captured["mode"] = tool_mode
        captured["count"] = len(tool_declarations)
        captured["return_none"] = return_none_when_no_tool
        return "done"

    monkeypatch.setattr(
        "app.ai.gemini_service.generate_gemini_function_call_reply", fake_gemini
    )

    out = await tool_loop.run_tool_loop("u1", "c1", "list my repositories", history=[])
    assert out == "done"
    assert captured["mode"] == "AUTO"
    assert captured["return_none"] is True
    assert captured["count"] == len(tool_loop.model_tool_declarations())
