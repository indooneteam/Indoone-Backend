"""User-scoped GitHub API operations.

OAuth token persistence remains shared with the central integration registry; all
GitHub provider calls and response normalization live in this plugin package.
"""

from __future__ import annotations

import re

import httpx

from app.capabilities.integrations import _fernet
from app.capabilities.store import get_integration_token

_GITHUB_BASE = "https://api.github.com"
_GITHUB_REPOSITORIES_URL = f"{_GITHUB_BASE}/user/repos"
_GITHUB_ISSUES_URL = f"{_GITHUB_BASE}/issues"
_GITHUB_PULLS_BASE_URL = f"{_GITHUB_BASE}/repos"
_GITHUB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def _github_headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _github_token(user_id: str) -> str:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    token_row = get_integration_token(user, "github")
    if token_row is None:
        raise ValueError("integration is not connected for user")
    try:
        return _fernet().decrypt(bytes(token_row["access_token"])).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("stored oauth token cannot be decrypted") from exc


def _pagination(page: int, per_page: int) -> dict[str, int]:
    if page < 1 or page > 1000:
        raise ValueError("page must be between 1 and 1000")
    if per_page < 1 or per_page > 100:
        raise ValueError("per_page must be between 1 and 100")
    return {"page": page, "per_page": per_page}


async def list_github_repositories(
    user_id: str, page: int = 1, per_page: int = 30
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    params = {**_pagination(page, per_page), "sort": "updated", "direction": "desc"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            _GITHUB_REPOSITORIES_URL,
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, list):
        raise RuntimeError("github provider returned invalid repository response")
    repositories: list[dict[str, object]] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        repositories.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "full_name": item.get("full_name"),
            "private": item.get("private"),
            "html_url": item.get("html_url"),
            "default_branch": item.get("default_branch"),
            "description": item.get("description"),
        })
    return {
        "integration": "github", "user_id": user, "page": page,
        "per_page": per_page, "repositories": repositories,
        "secrets_exposed": False,
    }


async def list_github_issues(
    user_id: str, page: int = 1, per_page: int = 30
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    params = {**_pagination(page, per_page), "state": "open", "filter": "assigned"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            _GITHUB_ISSUES_URL,
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, list):
        raise RuntimeError("github provider returned invalid issues response")
    issues: list[dict[str, object]] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        labels = item.get("labels")
        issues.append({
            "id": item.get("id"),
            "number": item.get("number"),
            "title": item.get("title"),
            "state": item.get("state"),
            "html_url": item.get("html_url"),
            "repository_url": item.get("repository_url"),
            "labels": [label.get("name") for label in labels if isinstance(label, dict)] if isinstance(labels, list) else [],
            "pull_request": item.get("pull_request"),
        })
    return {
        "integration": "github", "user_id": user, "page": page,
        "per_page": per_page, "issues": issues, "secrets_exposed": False,
    }


async def list_github_pull_requests(
    user_id: str, repository: str, page: int = 1, per_page: int = 30
) -> dict[str, object]:
    user = user_id.strip()
    repo = repository.strip()
    if not user:
        raise ValueError("user_id is required")
    if not _GITHUB_REPO_PATTERN.fullmatch(repo):
        raise ValueError("repository must use owner/name format")
    params = {**_pagination(page, per_page), "state": "open", "sort": "updated", "direction": "desc"}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_PULLS_BASE_URL}/{repo}/pulls",
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        response.raise_for_status()
        body = response.json()
    if not isinstance(body, list):
        raise RuntimeError("github provider returned invalid pull request response")
    pull_requests: list[dict[str, object]] = []
    for item in body:
        if not isinstance(item, dict):
            continue
        pull_requests.append({
            "id": item.get("id"), "number": item.get("number"),
            "title": item.get("title"), "state": item.get("state"),
            "html_url": item.get("html_url"), "draft": item.get("draft"),
            "head": item.get("head"), "base": item.get("base"),
        })
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "page": page, "per_page": per_page, "pull_requests": pull_requests,
        "secrets_exposed": False,
    }
