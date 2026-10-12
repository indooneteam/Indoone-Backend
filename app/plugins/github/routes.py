"""FastAPI routes for the GitHub integration plugin."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.github.service import (
    list_github_issues,
    list_github_pull_requests,
    list_github_repositories,
)
from app.plugins.github.writes import github_comment_issue_or_pr, github_create_issue

router = APIRouter(prefix="/github", tags=["github"])


class GithubRepositoriesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    page: int = Field(default=1, ge=1, le=1000)
    per_page: int = Field(default=30, ge=1, le=100)


class GithubIssuesRequest(GithubRepositoriesRequest):
    pass


class GithubPullRequestsRequest(GithubRepositoriesRequest):
    repository: str = Field(min_length=3, max_length=200)


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
        raise HTTPException(status_code=502, detail="GitHub provider request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/issues")
async def github_issues(request: GithubIssuesRequest) -> dict[str, object]:
    try:
        return await list_github_issues(request.user_id, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="GitHub provider request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/pull-requests")
async def github_pull_requests(request: GithubPullRequestsRequest) -> dict[str, object]:
    try:
        return await list_github_pull_requests(request.user_id, request.repository, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="GitHub provider request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/create-issue")
async def github_create_issue_endpoint(request: GithubCreateIssueRequest) -> dict[str, object]:
    try:
        return await github_create_issue(request.user_id, request.repository, request.title, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="GitHub provider request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/comment")
async def github_comment_endpoint(request: GithubCommentRequest) -> dict[str, object]:
    try:
        return await github_comment_issue_or_pr(request.user_id, request.repository, request.number, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="GitHub provider request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
