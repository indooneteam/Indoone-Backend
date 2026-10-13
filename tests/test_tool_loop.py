import json

import pytest

from app.ai import tool_loop
from app.ai.approval import issue_approval_token
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

    # Data-changing tools are offered too; approval is enforced at dispatch.
    assert "sandbox_execution" in names
    assert "github_update_repository_file" in names
    assert "phone_call" in names
    assert "search_web" in names
    assert "google_calendar_list_events" in names
    assert "google_calendar_create_event" in names
    assert "google_drive_list_files" in names
    assert "google_drive_upload_file" in names
    assert "google_photos_search_media" in names
    assert "youtube_search" in names
    assert "instagram_list_media" in names
    assert "facebook_list_pages" in names
    assert "canva_list_designs" in names
    assert "data_analysis" in names
    assert "task_schedule_build" in names

    # No duplicates, and every entry is fully described. No-argument tools
    # legitimately omit the parameter schema.
    assert len(names) == len(set(names))
    for item in declarations:
        assert item["description"]
        assert isinstance(item.get("parameters", {}), dict)


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
        "app.ai.approval.authorize_tool",
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
        "app.ai.gemini_service.generate_gemini_function_call_reply", fake_gemini
    )

    out = await tool_loop.run_tool_loop("u1", "c1", "list my repositories", history=[])
    assert out == "done"
    assert captured["mode"] == "AUTO"
    assert captured["return_none"] is True
    assert captured["reuse_text"] is True
    assert captured["count"] == len(tool_loop.model_tool_declarations("u1"))


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
