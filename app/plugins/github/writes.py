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
    create: bool = False,
) -> dict[str, object]:
    """Commit an approved text-file create/update without overwriting newer content."""
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
    if create:
        if normalized_sha:
            raise ValueError("new files must not specify an existing file SHA")
    elif not re.fullmatch(r"[0-9a-f]{40}", normalized_sha):
        raise ValueError("updating an existing file requires its current 40-character Git object SHA")
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
    }
    if not create:
        payload["sha"] = normalized_sha
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
                message = (
                    "That file path already exists or GitHub rejected the new path. Refresh the folder and choose a new file name."
                    if create
                    else "This file changed on GitHub after it was opened. Refresh the file and review the new version before committing."
                )
                raise GitHubRepositoryWriteConflict(message) from exc
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
        "operation": "create_file" if create else "update_file",
        "secrets_exposed": False,
    }


# ---------------------------------------------------------------------------
# The rest of the flow: start a branch, open a pull request, merge it, clean up.
# Every one of these goes through _validate, so approval is required and a
# repository is always owner/name. Nothing here acts without it.
# ---------------------------------------------------------------------------


async def github_create_branch(
    user_id: str,
    repository: str,
    branch: str,
    from_ref: str = "",
    approved: bool = False,
) -> dict[str, object]:
    """Create a branch from a ref, so a change can be made without touching main."""
    user, repo = _validate(user_id, repository, approved)
    branch_name = _validated_repository_ref(branch)
    if not branch_name:
        raise ValueError("branch is required")
    source = _validated_repository_ref(from_ref) or "HEAD"
    async with httpx.AsyncClient(timeout=20.0) as client:
        headers = _github_headers(_github_token(user))
        # A branch is created from a commit sha, so resolve the ref first.
        if re.fullmatch(r"[0-9a-fA-F]{40}", source):
            base_sha = source.lower()
        else:
            ref_response = await client.get(
                f"https://api.github.com/repos/{repo}/git/ref/heads/{quote(source, safe='')}",
                headers=headers,
            )
            ref_response.raise_for_status()
            ref_body = ref_response.json()
            ref_object = ref_body.get("object") if isinstance(ref_body, dict) else None
            ref_object = ref_object if isinstance(ref_object, dict) else {}
            base_sha = str(ref_object.get("sha") or "")
            if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
                raise RuntimeError("could not resolve the source branch to a commit")

        response = await client.post(
            f"https://api.github.com/repos/{repo}/git/refs",
            headers=headers,
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha},
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    ref_object = result.get("object")
    ref_object = ref_object if isinstance(ref_object, dict) else {}
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "operation": "create_branch",
        "result": {"branch": branch_name, "sha": str(ref_object.get("sha") or ""), "from": source},
        "secrets_exposed": False,
    }


async def github_delete_branch(
    user_id: str, repository: str, branch: str, approved: bool = False
) -> dict[str, object]:
    """Delete a branch - the clean-up after a pull request is merged."""
    user, repo = _validate(user_id, repository, approved)
    branch_name = _validated_repository_ref(branch)
    if not branch_name:
        raise ValueError("branch is required")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.delete(
            f"https://api.github.com/repos/{repo}/git/refs/heads/{quote(branch_name, safe='')}",
            headers=_github_headers(_github_token(user)),
        )
        response.raise_for_status()
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "operation": "delete_branch", "result": {"branch": branch_name, "deleted": True},
        "secrets_exposed": False,
    }


async def github_create_pull_request(
    user_id: str,
    repository: str,
    title: str,
    head: str,
    base: str = "main",
    body: str = "",
    approved: bool = False,
) -> dict[str, object]:
    """Open a pull request from head into base."""
    user, repo = _validate(user_id, repository, approved)
    head_ref = _validated_repository_ref(head)
    base_ref = _validated_repository_ref(base)
    if not head_ref or not base_ref:
        raise ValueError("head and base branches are required")
    if head_ref == base_ref:
        raise ValueError("head and base must be different branches")
    payload = {
        "title": _text(title, "title", 200),
        "head": head_ref,
        "base": base_ref,
        "body": body.strip()[:20000],
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{_GITHUB_BASE}/{repo}/pulls",
            headers=_github_headers(_github_token(user)),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "operation": "create_pull_request",
        "result": {
            "number": result.get("number"), "html_url": result.get("html_url"),
            "title": result.get("title"), "state": result.get("state"),
            "head": head_ref, "base": base_ref,
        },
        "secrets_exposed": False,
    }


async def github_merge_pull_request(
    user_id: str,
    repository: str,
    number: int,
    method: str = "squash",
    approved: bool = False,
) -> dict[str, object]:
    """Merge a pull request. Squash by default, as the team's own history is squashed."""
    user, repo = _validate(user_id, repository, approved)
    if number < 1 or number > 100000000:
        raise ValueError("pull request number is out of range")
    normalized_method = method.strip().lower() or "squash"
    if normalized_method not in {"squash", "merge", "rebase"}:
        raise ValueError("method must be squash, merge or rebase")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.put(
            f"{_GITHUB_BASE}/{repo}/pulls/{number}/merge",
            headers=_github_headers(_github_token(user)),
            json={"merge_method": normalized_method},
        )
        if response.status_code in {405, 409}:
            # Not mergeable: a conflict, or checks that have not passed. Say so
            # plainly rather than reporting a generic failure.
            raise RuntimeError(
                "the pull request cannot be merged yet - it may have conflicts "
                "or checks that have not passed"
            )
        response.raise_for_status()
        result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError("github provider returned invalid write response")
    return {
        "integration": "github", "user_id": user, "repository": repo,
        "operation": "merge_pull_request",
        "result": {
            "number": number, "merged": bool(result.get("merged")),
            "sha": str(result.get("sha") or ""), "message": str(result.get("message") or ""),
            "method": normalized_method,
        },
        "secrets_exposed": False,
    }
