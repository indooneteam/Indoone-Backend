"""Explicitly approved GitHub issue/comment write actions."""

from __future__ import annotations

import base64
import re
from urllib.parse import quote

import httpx

from app.plugins.github.service import (
    MAX_REPOSITORY_FILE_BYTES,
    _github_headers,
    _github_token,
    _redact_repository_secrets,
    _validated_repository,
    _validated_repository_path,
    _validated_repository_ref,
)

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
    return {
        "id": result.get("id"), "number": result.get("number"),
        "html_url": result.get("html_url"), "title": result.get("title"),
        "state": result.get("state"),
    }


async def github_create_issue(
    user_id: str, repository: str, title: str, body: str = "", approved: bool = False
) -> dict[str, object]:
    user, repo = _validate(user_id, repository, approved)
    payload = {"title": _text(title, "title", 2000), "body": body.strip()[:12000]}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GITHUB_BASE}/{repo}/issues",
            headers=_github_headers(_github_token(user)),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "operation": "create_issue", "result": _summary(result),
        "secrets_exposed": False,
    }


async def github_comment_issue_or_pr(
    user_id: str, repository: str, number: int, body: str, approved: bool = False
) -> dict[str, object]:
    user, repo = _validate(user_id, repository, approved)
    if number < 1 or number > 100000000:
        raise ValueError("number must be between 1 and 100000000")
    payload = {"body": _text(body, "body", 12000)}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GITHUB_BASE}/{repo}/issues/{number}/comments",
            headers=_github_headers(_github_token(user)),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "number": number, "operation": "comment", "result": _summary(result),
        "secrets_exposed": False,
    }

class GitHubRepositoryWriteConflict(RuntimeError):
    """Raised when the repository file changed since it was loaded."""


async def github_update_repository_file(
    user_id: str,
    repository: str,
    path: str,
    content: str,
    message: str,
    expected_sha: str,
    approval_verified: bool = False,
    branch: str = "",
) -> dict[str, object]:
    """Commit an explicitly approved text-file update without overwriting a newer version."""
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_path = _validated_repository_path(path)
    if not normalized_path:
        raise ValueError("path must identify a file")
    if not approval_verified:
        raise PermissionError("a valid, server-issued file-update approval is required")

    normalized_sha = expected_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", normalized_sha):
        raise ValueError("expected file SHA must be a 40-character Git object SHA")
    normalized_branch = _validated_repository_ref(branch)
    cleaned_message = _text(message, "commit message", 200)
    if not isinstance(content, str) or not content:
        raise ValueError("file content cannot be empty")
    content_bytes = content.encode("utf-8")
    if len(content_bytes) > MAX_REPOSITORY_FILE_BYTES:
        raise ValueError(f"file content exceeds the {MAX_REPOSITORY_FILE_BYTES}-byte limit")
    if content != _redact_repository_secrets(content):
        raise ValueError("file content appears to contain a credential or private key; remove it before committing")

    lower_path = normalized_path.casefold()
    filename = lower_path.rsplit("/", 1)[-1]
    if (
        filename.startswith(".env")
        or filename in {"credentials", "credentials.json", "id_rsa", "id_ed25519", "secrets.json"}
        or filename.endswith((".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"))
        or ".git/" in lower_path
    ):
        raise ValueError("refusing to commit a likely secret or private-key file")

    payload: dict[str, str] = {
        "message": cleaned_message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
        "sha": normalized_sha,
    }
    if normalized_branch:
        payload["branch"] = normalized_branch

    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.put(
            f"{_GITHUB_BASE}/{repo}/contents/{quote(normalized_path, safe='/')}",
            headers=_github_headers(_github_token(user)),
            json=payload,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {409, 422}:
                raise GitHubRepositoryWriteConflict(
                    "This file changed on GitHub after it was opened. Refresh the file and review the new version before committing."
                ) from exc
            raise
        body = response.json()

    if not isinstance(body, dict):
        raise RuntimeError("GitHub returned an invalid file-commit response")
    file_data = body.get("content")
    commit_data = body.get("commit")
    if not isinstance(file_data, dict) or not isinstance(commit_data, dict):
        raise RuntimeError("GitHub did not confirm the file commit")
    commit_sha = str(commit_data.get("sha") or "")
    file_sha = str(file_data.get("sha") or "")
    if not commit_sha or not file_sha:
        raise RuntimeError("GitHub returned an incomplete file-commit confirmation")
    return {
        "integration": "github",
        "repository": repo,
        "path": str(file_data.get("path") or normalized_path),
        "branch": normalized_branch,
        "file_sha": file_sha,
        "file_url": str(file_data.get("html_url") or ""),
        "commit_sha": commit_sha,
        "commit_url": str(commit_data.get("html_url") or ""),
        "commit_message": cleaned_message,
        "operation": "update_file",
        "secrets_exposed": False,
    }
