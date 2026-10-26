"""User-scoped GitHub API operations.

OAuth token persistence remains shared with the central integration registry; all
GitHub provider calls and response normalization live in this plugin package.
"""

from __future__ import annotations

import base64
import re

import httpx
from urllib.parse import quote

from app.capabilities.connectors.integrations import _fernet
from app.capabilities.connectors.store import get_integration_token

_GITHUB_BASE = "https://api.github.com"
_GITHUB_REPOSITORIES_URL = f"{_GITHUB_BASE}/user/repos"
_GITHUB_ISSUES_URL = f"{_GITHUB_BASE}/issues"
_GITHUB_PULLS_BASE_URL = f"{_GITHUB_BASE}/repos"
_GITHUB_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubAuthError(RuntimeError):
    """The stored GitHub credential is missing, expired, or was rejected."""


def _raise_github_error(response: "httpx.Response") -> None:
    """Turn a GitHub status into a message a user can act on.

    The plain raise_for_status produced "Client error '401 Unauthorized' for url ..." -
    an httpx sentence, repeated on every retry, that never said the token needed
    re-linking. These do.
    """
    status = int(getattr(response, "status_code", 0) or 0)

    if status < 400:
        return

    if status == 401:
        raise GitHubAuthError(
            "GitHub rejected the stored token (HTTP 401). The account is not linked "
            "properly or the token expired - re-authenticate GitHub and ask again. "
            "Nothing was read and nothing was changed."
        )

    if status == 403:
        raise GitHubAuthError(
            "GitHub refused the request (HTTP 403). The token is missing a scope this "
            "needs, or the rate limit is spent - check the token's repo scope and "
            "re-authenticate if needed. Nothing was changed."
        )

    if status == 404:
        raise RuntimeError(
            "GitHub returned 404: the repository, branch, pull request or file was not "
            "found with this token's access. Nothing was changed."
        )

    _raise_github_error(response)


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
        # Same sentence as the 401, because it is the same problem for the user - but a
        # ValueError, because "not connected yet" is a configuration answer the caller
        # reports as 400, not a server fault. A token GitHub actually rejected is the
        # GitHubAuthError above.
        raise ValueError(
            "GitHub is not connected for this account, or the token expired. "
            "Connect or re-authenticate GitHub and ask again. Nothing was changed."
        )
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
        _raise_github_error(response)
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
        _raise_github_error(response)
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
        _raise_github_error(response)
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


_GITHUB_REPOS_BASE_URL = f"{_GITHUB_BASE}/repos"
_GITHUB_CODE_SEARCH_URL = f"{_GITHUB_BASE}/search/code"
MAX_REPOSITORY_FILE_BYTES = 50_000
MAX_REPOSITORY_FILE_CHARS = 20_000
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?"
    r"-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.DOTALL,
)
_TOKEN_REDACTIONS = (
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_ACCESS_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"), "[REDACTED_API_KEY]"),
)


def _validated_repository(repository: str) -> str:
    normalized = repository.strip()
    if not _GITHUB_REPO_PATTERN.fullmatch(normalized):
        raise ValueError("repository must use owner/name format")
    return normalized


def _validated_repository_path(path: str) -> str:
    normalized = path.strip().strip("/")
    if len(normalized) > 1024 or chr(0) in normalized or chr(92) in normalized:
        raise ValueError("repository path is invalid")
    if any(segment in {".", ".."} for segment in normalized.split("/") if segment):
        raise ValueError("repository path cannot contain dot or parent segments")
    return normalized


def _validated_repository_ref(ref: str) -> str:
    normalized = ref.strip()
    if not normalized:
        return ""
    if (
        len(normalized) > 128
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", normalized)
        or normalized.startswith("/")
        or normalized.endswith("/")
        or ".." in normalized
    ):
        raise ValueError("branch or commit reference is invalid")
    return normalized


def _redact_repository_secrets(text: str) -> str:
    redacted = _PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", text)
    for pattern, replacement in _TOKEN_REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


async def list_github_repository_contents(
    user_id: str, repository: str, path: str = "", ref: str = ""
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_path = _validated_repository_path(path)
    normalized_ref = _validated_repository_ref(ref)
    path_suffix = f"/{quote(normalized_path, safe='/')}" if normalized_path else ""
    params = {"ref": normalized_ref} if normalized_ref else {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/contents{path_suffix}",
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        _raise_github_error(response)
        body = response.json()
    if isinstance(body, dict):
        if body.get("type") == "file":
            raise ValueError("the selected path is a file; use the GitHub file-read action")
        raise RuntimeError("GitHub returned an invalid repository-contents response")
    if not isinstance(body, list):
        raise RuntimeError("GitHub returned an invalid repository-contents response")
    contents: list[dict[str, object]] = []
    for item in body[:1000]:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type not in {"file", "dir", "symlink", "submodule"}:
            continue
        contents.append({
            "name": str(item.get("name") or ""),
            "path": str(item.get("path") or ""),
            "type": item_type,
            "size": int(item.get("size") or 0),
            "html_url": str(item.get("html_url") or ""),
        })
    return {
        "integration": "github", "repository": repo, "path": normalized_path,
        "ref": normalized_ref, "contents": contents, "count": len(contents),
        "secrets_exposed": False,
    }


async def read_github_repository_file(
    user_id: str, repository: str, path: str, ref: str = ""
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_path = _validated_repository_path(path)
    if not normalized_path:
        raise ValueError("path must identify a file")
    normalized_ref = _validated_repository_ref(ref)
    params = {"ref": normalized_ref} if normalized_ref else {}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/contents/{quote(normalized_path, safe='/')}",
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        _raise_github_error(response)
        body = response.json()
    if not isinstance(body, dict) or body.get("type") != "file":
        raise ValueError("the selected path is not a readable file")
    size = int(body.get("size") or 0)
    if size > MAX_REPOSITORY_FILE_BYTES:
        raise ValueError(
            f"file is {size} bytes; the AI file-read limit is {MAX_REPOSITORY_FILE_BYTES} bytes."
        )
    encoded = str(body.get("content") or "").strip()
    if not encoded:
        raise ValueError("GitHub did not return text content for this file")
    try:
        decoded = base64.b64decode(encoded, validate=False).decode("utf-8", errors="replace")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("GitHub returned an unreadable file encoding") from exc
    decoded = _redact_repository_secrets(decoded)
    truncated = len(decoded) > MAX_REPOSITORY_FILE_CHARS
    if truncated:
        decoded = decoded[:MAX_REPOSITORY_FILE_CHARS] + "\n[File output truncated by safety limit.]"
    return {
        "integration": "github", "repository": repo,
        "path": str(body.get("path") or normalized_path), "ref": normalized_ref,
        "size": size, "sha": str(body.get("sha") or ""),
        "html_url": str(body.get("html_url") or ""), "content": decoded,
        "truncated": truncated, "secrets_exposed": False,
    }


async def search_github_repository_code(
    user_id: str, repository: str, query: str, per_page: int = 10
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_query = query.strip()
    if not normalized_query or len(normalized_query) > 256:
        raise ValueError("search query must contain 1 through 256 characters")
    if per_page < 1 or per_page > 30:
        raise ValueError("per_page must be between 1 and 30")
    headers = _github_headers(_github_token(user))
    headers["Accept"] = "application/vnd.github.text-match+json"
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            _GITHUB_CODE_SEARCH_URL, headers=headers,
            params={"q": f"{normalized_query} repo:{repo}", "per_page": per_page},
        )
        _raise_github_error(response)
        body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("items"), list):
        raise RuntimeError("GitHub returned an invalid code-search response")
    results: list[dict[str, object]] = []
    for item in body["items"][:30]:
        if not isinstance(item, dict):
            continue
        matches = item.get("text_matches")
        snippets: list[str] = []
        if isinstance(matches, list):
            for match in matches[:3]:
                if isinstance(match, dict) and isinstance(match.get("fragment"), str):
                    snippets.append(_redact_repository_secrets(str(match["fragment"]))[:800])
        results.append({
            "name": str(item.get("name") or ""), "path": str(item.get("path") or ""),
            "html_url": str(item.get("html_url") or ""), "snippets": snippets,
        })
    return {
        "integration": "github", "repository": repo, "query": normalized_query,
        "results": results, "count": len(results), "secrets_exposed": False,
    }


async def list_github_repository_commits(
    user_id: str, repository: str, path: str = "", per_page: int = 10
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_path = _validated_repository_path(path)
    if per_page < 1 or per_page > 30:
        raise ValueError("per_page must be between 1 and 30")
    params: dict[str, object] = {"per_page": per_page}
    if normalized_path:
        params["path"] = normalized_path
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/commits",
            headers=_github_headers(_github_token(user)), params=params,
        )
        _raise_github_error(response)
        body = response.json()
    if isinstance(body, dict) and isinstance(body.get("items"), list):
        body = body["items"]
    if not isinstance(body, list):
        raise RuntimeError("GitHub returned an invalid commit-history response")
    commits: list[dict[str, object]] = []
    for item in body[:30]:
        if not isinstance(item, dict):
            continue
        commit = item.get("commit")
        commit = commit if isinstance(commit, dict) else {}
        author = commit.get("author")
        author = author if isinstance(author, dict) else {}
        # A commit can have an empty or missing message. "".splitlines() is [], and
        # indexing it raised IndexError, which the route turned into an unhandled 500.
        message_lines = str(commit.get("message") or "").splitlines()
        commits.append({
            "sha": str(item.get("sha") or ""),
            "message": message_lines[0][:300] if message_lines else "",
            "author": str(author.get("name") or ""), "date": str(author.get("date") or ""),
            "html_url": str(item.get("html_url") or ""),
        })
    return {
        "integration": "github", "repository": repo, "path": normalized_path,
        "commits": commits, "count": len(commits), "secrets_exposed": False,
    }


async def compare_github_repository_commits(
    user_id: str, repository: str, base: str, head: str
) -> dict[str, object]:
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    base_ref = _validated_repository_ref(base)
    head_ref = _validated_repository_ref(head)
    if not base_ref or not head_ref:
        raise ValueError("base and head references are required")
    compare_path = (
        f"{_GITHUB_REPOS_BASE_URL}/{repo}/compare/"
        f"{quote(base_ref, safe='/-._')}...{quote(head_ref, safe='/-._')}"
    )
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(compare_path, headers=_github_headers(_github_token(user)))
        _raise_github_error(response)
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("GitHub returned an invalid commit-comparison response")
    files: list[dict[str, object]] = []
    items = body.get("files")
    if isinstance(items, list):
        for item in items[:100]:
            if not isinstance(item, dict):
                continue
            files.append({
                "filename": str(item.get("filename") or ""),
                "status": str(item.get("status") or ""),
                "additions": int(item.get("additions") or 0),
                "deletions": int(item.get("deletions") or 0),
                "changes": int(item.get("changes") or 0),
                "html_url": str(item.get("blob_url") or ""),
            })
    return {
        "integration": "github", "repository": repo, "base": base_ref, "head": head_ref,
        "status": str(body.get("status") or ""), "ahead_by": int(body.get("ahead_by") or 0),
        "behind_by": int(body.get("behind_by") or 0), "total_commits": int(body.get("total_commits") or 0),
        "files": files, "secrets_exposed": False,
    }


path = "app/plugins/github/service.py"
message = "Fix GitHub API response robustness and commit-history fallback handling"


# ---------------------------------------------------------------------------
# The rest of the flow: branches, a pull request's files, checks, one commit.
# Same shape as every read above: validate, call, normalise, never leak secrets.
# ---------------------------------------------------------------------------


# A repository with more branches than this is not a real branch list any more; the
# read stops and says so rather than paging forever.
_MAX_BRANCHES = 1000


async def list_github_branches(
    user_id: str, repository: str, per_page: int = 30
) -> dict[str, object]:
    """List a repository's branches, so work can be started from the right one."""
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    if per_page < 1 or per_page > 100:
        raise ValueError("per_page must be between 1 and 100")
    # Every page, not the first one. The list used to stop at per_page (30 by default),
    # so a repository with 40 branches was reported as having 30 - a wrong number stated
    # with confidence, which is worse than an error. The cap is a guard against a
    # pathological repository, and it is reported rather than hidden.
    headers = _github_headers(_github_token(user))
    body: list[object] = []
    page = 1
    # Stopping at the cap is not the same as reaching the end: the loop has to say which
    # one happened, or the count reads as complete when it is not.
    truncated = False

    async with httpx.AsyncClient(timeout=20.0) as client:
        while True:
            if len(body) >= _MAX_BRANCHES:
                truncated = True
                break

            response = await client.get(
                f"{_GITHUB_REPOS_BASE_URL}/{repo}/branches",
                headers=headers,
                params={"per_page": per_page, "page": page},
            )
            _raise_github_error(response)
            batch = response.json()

            if not isinstance(batch, list):
                raise RuntimeError("GitHub returned an invalid branch list")

            body.extend(batch)

            # A short page is the last page.
            if len(batch) < per_page:
                break

            page += 1

    body = body[:_MAX_BRANCHES]
    branches: list[dict[str, object]] = []
    for item in body[:100]:
        if not isinstance(item, dict):
            continue
        commit = item.get("commit")
        commit = commit if isinstance(commit, dict) else {}
        branches.append({
            "name": str(item.get("name") or ""),
            "sha": str(commit.get("sha") or ""),
            "protected": bool(item.get("protected")),
        })
    return {
        "integration": "github", "repository": repo,
        "branches": branches, "count": len(branches),
        # True only when the cap stopped the read, so the model can say "the first N"
        # instead of claiming that is all there is.
        "truncated": truncated, "secrets_exposed": False,
    }


async def list_github_pull_request_files(
    user_id: str, repository: str, number: int, per_page: int = 30
) -> dict[str, object]:
    """The diff of a pull request - which files changed, and the patch for each."""
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    if number < 1 or number > 100000000:
        raise ValueError("pull request number is out of range")
    if per_page < 1 or per_page > 100:
        raise ValueError("per_page must be between 1 and 100")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/pulls/{number}/files",
            headers=_github_headers(_github_token(user)),
            params={"per_page": per_page},
        )
        _raise_github_error(response)
        body = response.json()
    if not isinstance(body, list):
        raise RuntimeError("GitHub returned an invalid pull-request file list")
    files: list[dict[str, object]] = []
    for item in body[:100]:
        if not isinstance(item, dict):
            continue
        files.append({
            "filename": str(item.get("filename") or ""),
            "status": str(item.get("status") or ""),
            "additions": int(item.get("additions") or 0),
            "deletions": int(item.get("deletions") or 0),
            "changes": int(item.get("changes") or 0),
            # The patch is the point of this call, and it can hold code, not
            # credentials - but it is still passed through the redactor.
            "patch": _redact_repository_secrets(str(item.get("patch") or ""))[:20000],
        })
    return {
        "integration": "github", "repository": repo, "number": number,
        "files": files, "count": len(files), "secrets_exposed": False,
    }


async def list_github_check_runs(
    user_id: str, repository: str, ref: str = ""
) -> dict[str, object]:
    """The CI status for a ref, so a change is not merged before it is green."""
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_ref = _validated_repository_ref(ref)
    params: dict[str, object] = {"per_page": 50}
    if normalized_ref:
        params["ref"] = normalized_ref
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/commits/{quote(normalized_ref or 'HEAD', safe='')}/check-runs"
            if normalized_ref else f"{_GITHUB_REPOS_BASE_URL}/{repo}/check-runs",
            headers=_github_headers(_github_token(user)),
            params=params,
        )
        if response.status_code == 404:
            return {
                "integration": "github", "repository": repo, "ref": normalized_ref,
                "check_runs": [], "count": 0, "conclusion": "none", "secrets_exposed": False,
            }
        _raise_github_error(response)
        body = response.json()
    if isinstance(body, dict):
        items = body.get("check_runs")
    else:
        items = body
    items = items if isinstance(items, list) else []
    runs: list[dict[str, object]] = []
    for item in items[:50]:
        if not isinstance(item, dict):
            continue
        runs.append({
            "name": str(item.get("name") or ""),
            "status": str(item.get("status") or ""),
            "conclusion": str(item.get("conclusion") or ""),
            "html_url": str(item.get("html_url") or ""),
        })
    conclusions = {str(r.get("conclusion") or "") for r in runs}
    if not runs:
        overall = "none"
    elif conclusions & {"failure", "timed_out", "cancelled", "action_required"}:
        overall = "failure"
    elif conclusions & {"", "neutral", "queued", "in_progress"}:
        overall = "running"
    else:
        overall = "success"
    return {
        "integration": "github", "repository": repo, "ref": normalized_ref,
        "check_runs": runs, "count": len(runs), "conclusion": overall,
        "secrets_exposed": False,
    }


async def get_github_commit(
    user_id: str, repository: str, sha: str
) -> dict[str, object]:
    """One commit, with its message and the files it touched."""
    user = user_id.strip()
    if not user:
        raise ValueError("user_id is required")
    repo = _validated_repository(repository)
    normalized_sha = sha.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", normalized_sha):
        raise ValueError("sha must be a 7 to 40 character hex commit id")
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(
            f"{_GITHUB_REPOS_BASE_URL}/{repo}/commits/{quote(normalized_sha, safe='')}",
            headers=_github_headers(_github_token(user)),
        )
        _raise_github_error(response)
        body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("GitHub returned an invalid commit response")
    commit = body.get("commit")
    commit = commit if isinstance(commit, dict) else {}
    author = commit.get("author")
    author = author if isinstance(author, dict) else {}
    message_lines = str(commit.get("message") or "").splitlines()
    files: list[dict[str, object]] = []
    for item in (body.get("files") if isinstance(body.get("files"), list) else [])[:100]:
        if not isinstance(item, dict):
            continue
        files.append({
            "filename": str(item.get("filename") or ""),
            "status": str(item.get("status") or ""),
            "additions": int(item.get("additions") or 0),
            "deletions": int(item.get("deletions") or 0),
            "patch": _redact_repository_secrets(str(item.get("patch") or ""))[:20000],
        })
    return {
        "integration": "github", "repository": repo,
        "sha": str(body.get("sha") or ""),
        "message": "\n".join(message_lines)[:4000],
        "author": str(author.get("name") or ""),
        "date": str(author.get("date") or ""),
        "html_url": str(body.get("html_url") or ""),
        "files": files, "count": len(files), "secrets_exposed": False,
    }
