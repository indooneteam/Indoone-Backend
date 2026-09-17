from __future__ import annotations

import re

import httpx

from app.capabilities.integrations import _github_headers, _github_token

_GITHUB_BASE = "https://api.github.com/repos"
_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _validate(user_id: str, repository: str, approved: bool) -> tuple[str, str]:
    user = user_id.strip()
    repo = repository.strip()
    if not user:
        raise ValueError("user_id is required")
    if not _REPO_PATTERN.fullmatch(repo):
        raise ValueError("repository must use owner/name format")
    if not approved:
        raise PermissionError("explicit approval is required for github write operations")
    return user, repo


def _text(value: str, field: str, max_length: int) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field} is required")
    if len(cleaned) > max_length:
        raise ValueError(f"{field} exceeds maximum length of {max_length}")
    return cleaned


def _summary(result: dict[str, object]) -> dict[str, object]:
    return {"id": result.get("id"), "number": result.get("number"), "html_url": result.get("html_url"), "title": result.get("title"), "state": result.get("state")}


async def github_create_issue(user_id: str, repository: str, title: str, body: str = "", approved: bool = False) -> dict[str, object]:
    user, repo = _validate(user_id, repository, approved)
    payload = {"title": _text(title, "title", 2000), "body": body.strip()[:12000]}
    access_token = _github_token(user)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{_GITHUB_BASE}/{repo}/issues", headers=_github_headers(access_token), json=payload)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {"integration": "github", "user_id": user, "repository": repo, "operation": "create_issue", "result": _summary(result), "secrets_exposed": False}


async def github_comment_issue_or_pr(user_id: str, repository: str, number: int, body: str, approved: bool = False) -> dict[str, object]:
    user, repo = _validate(user_id, repository, approved)
    if number < 1 or number > 100000000:
        raise ValueError("number must be between 1 and 100000000")
    payload = {"body": _text(body, "body", 12000)}
    access_token = _github_token(user)
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{_GITHUB_BASE}/{repo}/issues/{number}/comments", headers=_github_headers(access_token), json=payload)
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {"integration": "github", "user_id": user, "repository": repo, "number": number, "operation": "comment", "result": _summary(result), "secrets_exposed": False}
