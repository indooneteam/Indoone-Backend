"""FastAPI routes for the GitHub integration plugin."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from app.ai.execution.approval import validate_approval_token
from app.ai.execution.connector_authorization import authorize_tool
from pydantic import BaseModel, Field
from app.api.security.errors import provider_failure

from app.plugins.github.service import (
    list_github_issues,
    list_github_pull_requests,
    list_github_repositories,
    list_github_repository_contents,
    read_github_repository_file,
    search_github_repository_code,
    list_github_repository_commits,
    compare_github_repository_commits,
)
from app.plugins.github.writes import (
    GitHubRepositoryWriteConflict,
    github_comment_issue_or_pr,
    github_create_issue,
    github_update_repository_file,
)

router = APIRouter(prefix="/github", tags=["github"])


class GithubRepositoriesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    page: int = Field(default=1, ge=1, le=1000)
    per_page: int = Field(default=30, ge=1, le=100)


class GithubIssuesRequest(GithubRepositoriesRequest):
    pass


class GithubPullRequestsRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)


class GithubRepositoryPathRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)
    path: str = Field(default="", max_length=1024)
    ref: str = Field(default="", max_length=128)


class GithubReadFileRequest(GithubRepositoryPathRequest):
    path: str = Field(min_length=1, max_length=1024)


class GithubCodeSearchRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)
    query: str = Field(min_length=1, max_length=256)
    per_page: int = Field(default=10, ge=1, le=30)


class GithubCommitHistoryRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)
    path: str = Field(default="", max_length=1024)
    per_page: int = Field(default=10, ge=1, le=30)


class GithubCompareCommitsRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)
    base: str = Field(min_length=1, max_length=128)
    head: str = Field(min_length=1, max_length=128)


class GithubCreateIssueRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    repository: str = Field(min_length=3, max_length=200)
    title: str = Field(min_length=1, max_length=2000)
    body: str = Field(default="", max_length=12000)
    approved: bool = False


class GithubCommentRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    repository: str = Field(min_length=3, max_length=200)
    number: int = Field(ge=1, le=100000000)
    body: str = Field(min_length=1, max_length=12000)
    approved: bool = False


@router.post("/repositories")
async def github_repositories(request: GithubRepositoriesRequest) -> dict[str, object]:
    try:
        return await list_github_repositories(request.user_id, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub provider request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/issues")
async def github_issues(request: GithubIssuesRequest) -> dict[str, object]:
    try:
        return await list_github_issues(request.user_id, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub provider request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/pull-requests")
async def github_pull_requests(request: GithubPullRequestsRequest) -> dict[str, object]:
    try:
        return await list_github_pull_requests(request.user_id, request.repository, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub provider request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/create-issue")
async def github_create_issue_endpoint(request: GithubCreateIssueRequest) -> dict[str, object]:
    try:
        return await github_create_issue(request.user_id, request.repository, request.title, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub provider request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/comment")
async def github_comment_endpoint(request: GithubCommentRequest) -> dict[str, object]:
    try:
        return await github_comment_issue_or_pr(request.user_id, request.repository, request.number, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub provider request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc




class GithubUpdateFileRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    repository: str = Field(min_length=3, max_length=200)
    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(min_length=1, max_length=50_000)
    message: str = Field(min_length=1, max_length=200)
    sha: str = Field(default="", max_length=40)
    create: bool = False
    branch: str = Field(default="", max_length=128)
    approval_token: str = Field(min_length=16, max_length=2048)


@router.post("/contents")
async def github_repository_contents(request: GithubRepositoryPathRequest) -> dict[str, object]:
    try:
        return await list_github_repository_contents(request.user_id, request.repository, request.path, request.ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub repository contents request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/read-file")
async def github_repository_read_file(request: GithubReadFileRequest) -> dict[str, object]:
    try:
        return await read_github_repository_file(request.user_id, request.repository, request.path, request.ref)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub repository file request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/search-code")
async def github_repository_search_code(request: GithubCodeSearchRequest) -> dict[str, object]:
    try:
        return await search_github_repository_code(request.user_id, request.repository, request.query, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub code search request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/commits")
async def github_repository_commits(request: GithubCommitHistoryRequest) -> dict[str, object]:
    try:
        return await list_github_repository_commits(request.user_id, request.repository, request.path, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub commit history request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc


@router.post("/compare")
async def github_repository_compare(request: GithubCompareCommitsRequest) -> dict[str, object]:
    try:
        return await compare_github_repository_commits(request.user_id, request.repository, request.base, request.head)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub commit comparison request", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc

@router.post("/commit-file")
async def github_repository_commit_file(request: GithubUpdateFileRequest) -> dict[str, object]:
    tool_name = "github_update_repository_file"
    decision = authorize_tool(request.user_id, tool_name)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)
    if not validate_approval_token(request.approval_token, request.user_id, tool_name):
        raise HTTPException(status_code=403, detail="A current approval is required before committing this file.")
    try:
        return await github_update_repository_file(
            user_id=request.user_id,
            repository=request.repository,
            path=request.path,
            content=request.content,
            message=request.message,
            expected_sha=request.sha,
            approval_verified=True,
            branch=request.branch,
            create=request.create,
        )
    except GitHubRepositoryWriteConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 403:
            raise HTTPException(
                status_code=403,
                detail=(
                    "The linked account does not have permission to write here. "
                    "Check the account's repository permissions."
                ),
            ) from exc
        raise provider_failure("GitHub could not commit the file update.", exc) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("GitHub file commit request failed.", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("github plugin", exc, status_code=503) from exc
