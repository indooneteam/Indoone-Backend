"""Agent 11 - the GitHub supervisor.

The model asks; this decides. Everything here is about the boundary: what may be called,
with what arguments, and what is allowed to come back into a prompt.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agents import load

supervisor = load("11_github_supervisor")


def run(call, **kwargs):
    return asyncio.run(supervisor.validate_and_execute(call, **kwargs))


# ---------------------------------------------------------------- 1. declarations


def test_the_declarations_are_function_calling_shaped() -> None:
    declarations = supervisor.get_tool_declarations()

    assert [item["name"] for item in declarations] == [
        "github_list_branches",
        "github_get_diff",
        "github_list_prs",
        "github_get_commit",
    ]

    for item in declarations:
        assert item["description"]
        assert item["parameters"]["type"] == "object"
        assert item["parameters"]["required"]
        for name, schema in item["parameters"]["properties"].items():
            assert schema["type"] in {"string", "integer"}


def test_a_caller_cannot_change_the_declarations() -> None:
    declarations = supervisor.get_tool_declarations()
    declarations[0]["name"] = "something_else"

    assert supervisor.get_tool_declarations()[0]["name"] == "github_list_branches"


# ---------------------------------------------------------------- 2. arguments


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("backend", "indooneteam/Indoone-Backend"),
        ("Indoone-Backend", "indooneteam/Indoone-Backend"),
        ("client", "indooneteam/Indoone-Client"),
        ("indooneteam/Indoone-Backend", "indooneteam/Indoone-Backend"),
    ],
)
def test_a_shorthand_resolves_to_a_repository(given: str, expected: str) -> None:
    arguments, problem = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_list_branches", {"repo": given})
    )

    assert problem is None
    assert arguments["repo"] == expected


@pytest.mark.parametrize(
    "ref",
    ["main; rm -rf /", "main && curl evil", "branch`whoami`", "main..head", "a b", "$(id)"],
)
def test_a_dangerous_ref_never_becomes_a_request(ref: str) -> None:
    arguments, problem = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_get_diff", {"repo": "backend", "base": ref, "head": "main"})
    )

    assert arguments is None
    assert "invalid git ref" in problem


def test_a_normal_ref_is_accepted() -> None:
    arguments, problem = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_get_diff", {"repo": "backend", "base": "main", "head": "feat/auth"})
    )

    assert problem is None
    assert arguments["head"] == "feat/auth"


def test_an_unknown_tool_is_refused() -> None:
    arguments, problem = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_delete_repository", {"repo": "backend"})
    )

    assert arguments is None
    assert "unknown tool" in problem


def test_a_missing_argument_is_refused() -> None:
    arguments, problem = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_get_diff", {"repo": "backend", "base": "main"})
    )

    assert arguments is None
    assert "missing argument" in problem


def test_a_limit_is_clamped_rather_than_trusted() -> None:
    arguments, _ = supervisor.validate_arguments(
        supervisor.GitHubToolCall("github_list_branches", {"repo": "backend", "limit": "9999"})
    )

    assert arguments["limit"] == 50


# ---------------------------------------------------------------- 3. the scrubber


@pytest.mark.parametrize(
    "secret",
    [
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "DATABASE_PASSWORD=hunter2hunter2",
        "postgres://admin:s3cr3tp4ss@db.example.com:5432/indoone",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    ],
)
def test_a_secret_in_a_diff_never_reaches_the_model(secret: str) -> None:
    diff = f"diff --git a/.env b/.env\n+{secret}\n"

    cleaned = supervisor.sanitize_payload(diff)

    assert "hunter2" not in cleaned
    assert "MIIEowIBAAKCAQEA" not in cleaned
    assert "abcdefghijklmnopqrstuvwxyz123456" not in cleaned
    assert "s3cr3tp4ss" not in cleaned
    assert "ghp_" not in cleaned


def test_an_ordinary_diff_is_left_readable() -> None:
    diff = "diff --git a/app/main.py b/app/main.py\n+def handler(request):\n+    return 200\n"

    assert supervisor.sanitize_payload(diff) == diff


def test_a_huge_diff_is_truncated_with_a_notice() -> None:
    diff = "line of a large diff\n" * 2000

    cleaned = supervisor.sanitize_payload(diff)

    assert len(cleaned) < supervisor.MAX_DIFF_CHARS + 200
    assert "[diff truncated" in cleaned
    assert "characters]" in cleaned


# ---------------------------------------------------------------- 4. failures


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "UNAUTHORIZED"),
        (403, "FORBIDDEN"),
        (404, "NOT_FOUND"),
        (429, "RATE_LIMITED"),
        (500, "UNAVAILABLE"),
        (503, "UNAVAILABLE"),
    ],
)
def test_every_http_failure_becomes_a_result_not_an_exception(status: int, code: str) -> None:
    class Boom(Exception):
        status_code = status

    async def executor(tool, arguments, token):
        raise Boom(f"HTTP {status}")

    result = run(supervisor.GitHubToolCall("github_list_branches", {"repo": "backend"}), executor=executor)

    assert result.success is False
    assert result.error_code == code
    assert result.error_message
    assert result.sanitized_output == ""


def test_the_failure_speaks_the_users_language() -> None:
    class Boom(Exception):
        status_code = 401

    async def executor(tool, arguments, token):
        raise Boom("unauthorized")

    result = run(
        supervisor.GitHubToolCall("github_list_branches", {"repo": "backend"}),
        executor=executor,
        language="kn",
    )

    assert result.error_code == "UNAUTHORIZED"
    assert result.error_message != supervisor._MESSAGES["UNAUTHORIZED"]["en"]


# ---------------------------------------------------------------- 5. success


def test_a_branch_lookup_returns_chips_and_clean_output() -> None:
    async def executor(tool, arguments, token):
        return {"branches": [{"name": "feat/auth"}, {"name": "main"}]}

    result = run(supervisor.GitHubToolCall("github_list_branches", {"repo": "backend"}), executor=executor)

    assert result.success is True
    assert "feat/auth" in result.sanitized_output
    assert result.action_chips == [{"label": "View Diff", "action": "diff"}]


def test_the_token_never_appears_in_the_result() -> None:
    async def executor(tool, arguments, token):
        return {"branches": []}

    result = run(
        supervisor.GitHubToolCall("github_list_branches", {"repo": "backend"}),
        token="ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        executor=executor,
    )

    assert "ghp_" not in result.sanitized_output
    assert "ghp_" not in str(result.data)


# ---------------------------------------------------------------- 6. speed


def test_validation_and_sanitizing_stay_under_the_budget() -> None:
    call = supervisor.GitHubToolCall("github_get_diff", {"repo": "backend", "base": "main", "head": "feat/auth"})
    diff = "diff --git a/x.py b/x.py\n+pass\n" * 50
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        supervisor.validate_arguments(call)
        supervisor.sanitize_payload(diff)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
