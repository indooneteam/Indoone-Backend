import json

import pytest

from app.ai.execution.tool_registry import get_tool_spec, requires_approval
from app.ai.execution.tools import ToolResult
from app.plugins.github import chat_tools


@pytest.mark.asyncio
async def test_github_repository_chat_uses_authenticated_read_tool(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")
    calls = []

    async def fake_run_tool_async(name, payload, **kwargs):
        calls.append((name, json.loads(payload), kwargs))
        return ToolResult(
            name=name,
            output=json.dumps(
                {
                    "repositories": [
                        {
                            "full_name": "indooneteam/indoone",
                            "private": False,
                            "default_branch": "main",
                        }
                    ],
                    "secrets_exposed": False,
                }
            ),
            safe=True,
        )

    monkeypatch.setattr(chat_tools, "run_tool_async", fake_run_tool_async)

    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "Show my latest GitHub repositories"
    )

    assert reply is not None
    assert "indooneteam/indoone" in reply
    assert "1. indooneteam/indoone (main)" in reply
    assert calls[0][0] == "github_list_repositories"
    assert calls[0][1]["user_id"] == "user-123"
    assert calls[0][2]["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_github_pull_requests_requires_explicit_owner_repository(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")

    async def unused_tool(*_args, **_kwargs):
        raise AssertionError("No GitHub request should execute without a repository.")

    monkeypatch.setattr(chat_tools, "run_tool_async", unused_tool)

    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "Show open GitHub pull requests"
    )

    assert reply is not None
    assert "owner/repository" in reply


@pytest.mark.asyncio
async def test_unrelated_chat_message_does_not_call_github(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")

    async def unused_tool(*_args, **_kwargs):
        raise AssertionError("An unrelated message must not execute a GitHub tool.")

    monkeypatch.setattr(chat_tools, "run_tool_async", unused_tool)

    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "Tell me a short story about the moon"
    )

    assert reply is None


@pytest.mark.asyncio
async def test_github_tool_error_does_not_claim_live_data(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")

    async def disconnected_tool(name, _payload, **_kwargs):
        return ToolResult(name=name, output="Tool error: connector is not connected for this user", safe=False)

    monkeypatch.setattr(chat_tools, "run_tool_async", disconnected_tool)

    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "List my GitHub repositories"
    )

    assert reply is not None
    assert "not connected" in reply.lower()
    assert "repository" not in reply.lower()


def test_github_ai_tools_are_read_only_connector_tools() -> None:
    for name, capability in (
        ("github_list_repositories", "repos.read"),
        ("github_list_assigned_issues", "issues.read"),
        ("github_list_pull_requests", "pull_requests.read"),
    ):
        spec = get_tool_spec(name)
        assert spec is not None
        assert spec.connector_id == "github"
        assert spec.capability == capability
        assert requires_approval(name) is False



@pytest.mark.asyncio
async def test_repository_inside_request_lists_actual_contents(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")
    calls = []

    async def fake_run_tool_async(name, payload, **kwargs):
        data = json.loads(payload)
        calls.append((name, data))
        if name == "github_list_repositories":
            return ToolResult(name=name, output=json.dumps({
                "repositories": [{"full_name": "indooneteam/indoone", "name": "indoone", "private": False, "default_branch": "main"}]
            }), safe=True)
        return ToolResult(name=name, output=json.dumps({
            "repository": "indooneteam/indoone", "path": "",
            "contents": [
                {"name": "app", "path": "app", "type": "dir", "size": 0},
                {"name": "README.md", "path": "README.md", "type": "file", "size": 50},
            ],
        }), safe=True)

    monkeypatch.setattr(chat_tools, "run_tool_async", fake_run_tool_async)
    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "What files are inside my GitHub repository indoone?"
    )
    assert reply is not None
    assert "README.md" in reply
    assert any(name == "github_list_repository_contents" for name, _ in calls)


@pytest.mark.asyncio
async def test_repository_file_request_routes_to_file_reader(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")
    calls = []

    async def fake_run_tool_async(name, payload, **kwargs):
        data = json.loads(payload)
        calls.append((name, data))
        if name == "github_list_repositories":
            return ToolResult(name=name, output=json.dumps({
                "repositories": [{"full_name": "indooneteam/indoone", "name": "indoone"}]
            }), safe=True)
        return ToolResult(name=name, output=json.dumps({
            "repository": "indooneteam/indoone", "path": "README.md",
            "content": "# Indoone\\nRead-only GitHub integration", "truncated": False,
        }), safe=True)

    monkeypatch.setattr(chat_tools, "run_tool_async", fake_run_tool_async)
    reply = await chat_tools.handle_github_chat_with_tools(
        "user-123", "Read README.md in my GitHub repo indoone"
    )
    assert reply is not None
    assert "# Indoone" in reply
    assert any(name == "github_read_repository_file" and data.get("path") == "README.md" for name, data in calls)


@pytest.mark.asyncio
async def test_file_request_without_path_asks_for_clarification(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_MODEL_BACKEND", "local")

    async def unused_tool(*_args, **_kwargs):
        raise AssertionError("No file tool should execute without a requested path.")

    monkeypatch.setattr(chat_tools, "run_tool_async", unused_tool)
    reply = await chat_tools.handle_github_chat_with_tools("user-123", "Read a file in my GitHub repo")
    assert reply is not None
    assert "which file" in reply.lower()

@pytest.mark.parametrize(
    ("message", "expected_hint"),
    [
        ("GitHub repo holage en ide", ""),
        ("nanna GitHub repo holage en ide", ""),
        ("Indoone repo holage en ide", "Indoone"),
        ("Indoone repo olage en ide", "Indoone"),
    ],
)
def test_romanized_kannada_repository_hint_ignores_intent_words(
    message: str, expected_hint: str
) -> None:
    assert chat_tools._repository_hint_from_message(message) == expected_hint


@pytest.mark.parametrize(
    ("message", "expected_hint"),
    [
        ("GitHub repo holage en ide", ""),
        ("nanna GitHub repo holage en ide", ""),
        ("Indoone repo holage en ide", "Indoone"),
    ],
)
def test_romanized_kannada_repo_contents_request_uses_github_reader(
    message: str, expected_hint: str
) -> None:
    request = chat_tools._github_action(message, None)
    assert isinstance(request, tuple)
    tool_name, arguments = request
    assert tool_name == "github_list_repository_contents"
    assert arguments["repository_hint"] == expected_hint

@pytest.mark.parametrize(
    ("message", "expected_hint"),
    [
        ("Nanna Indoone repo holage en ede", "Indoone"),
        ("Nanna Indoone repo olage en ide", "Indoone"),
        ("Indoone repository holagade en ide", "Indoone"),
        ("Indoone repo en ede", "Indoone"),
    ],
)
def test_romanized_kannada_repository_contents_variants_are_routed(
    message: str, expected_hint: str,
) -> None:
    action = chat_tools._github_action(message, [])
    assert isinstance(action, tuple)
    assert action[0] == "github_list_repository_contents"
    assert action[1]["repository_hint"] == expected_hint


@pytest.mark.parametrize(
    ("message", "expected_hint"),
    [
        ("ನನ್ನ GitHub ರೆಪೊಸಿಟರಿಯ ಒಳಗಿರುವ ಫೈಲ್‌ಗಳ ಪಟ್ಟಿ ತೋರಿಸು", ""),
        ("ನನ್ನ Indoone ರೆಪೊಸಿಟರಿಯ ಒಳಗಿರುವ ಫೈಲ್‌ಗಳ ಪಟ್ಟಿ ತೋರಿಸು", "Indoone"),
        ("GitHub Indoone repository ಒಳಗೆ ಏನೆಲ್ಲಾ ಇದೆ ತೋರಿಸು", "Indoone"),
        ("ನನ್ನ Indoone ರೆಪೊಸಿಟರಿಯಲ್ಲಿ ಏನೇನು ಫೈಲ್‌ಗಳಿವೆ", "Indoone"),
    ],
)
def test_kannada_repository_contents_requests_route_to_github_reader(
    message: str, expected_hint: str,
) -> None:
    action = chat_tools._github_action(message, [])
    assert isinstance(action, tuple)
    assert action[0] == "github_list_repository_contents"
    assert action[1]["repository_hint"] == expected_hint

@pytest.mark.asyncio
async def test_repository_hint_prefers_exact_name_over_substring(monkeypatch) -> None:
    calls = []

    async def fake_run_tool_async(name, payload, **kwargs):
        data = json.loads(payload)
        calls.append((name, data))
        if name == "github_list_repositories":
            return ToolResult(name=name, output=json.dumps({
                "repositories": [
                    {"full_name": "indooneteam/Indoone-Backend", "name": "Indoone-Backend"},
                    {"full_name": "indooneteam/indoone", "name": "indoone"},
                ]
            }), safe=True)
        return ToolResult(name=name, output=json.dumps({
            "repository": data.get("repository"), "path": data.get("path", ""),
            "contents": [], "count": 0, "secrets_exposed": False,
        }), safe=True)

    monkeypatch.setattr(chat_tools, "run_tool_async", fake_run_tool_async)
    result = await chat_tools._execute_github_tool(
        "user-123", "github_list_repository_contents",
        {"repository_hint": "Indoone", "path": ""},
    )

    assert result["ok"] is True
    assert result["result"]["repository"] == "indooneteam/indoone"
    assert any(
        name == "github_list_repository_contents" and data.get("repository") == "indooneteam/indoone"
        for name, data in calls
    )
