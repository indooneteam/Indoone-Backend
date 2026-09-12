from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.github_writes import github_comment_issue_or_pr, github_create_issue
from app.capabilities.gmail import get_gmail_message, list_gmail_messages, send_gmail_message
from app.capabilities.integrations import (
    list_github_issues,
    list_github_pull_requests,
    list_github_repositories,
    probe_integration,
)

router = APIRouter(prefix="/integrations", tags=["integrations"])


class IntegrationProbeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


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


class GmailMessagesRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    query: str = Field(default="", max_length=500)
    page_token: str = Field(default="", max_length=2048)
    max_results: int = Field(default=20, ge=1, le=100)


class GmailMessageRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)


class GmailSendRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    to: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=2000)
    body: str = Field(min_length=1, max_length=20000)
    approved: bool = False


@router.post("/{integration_id}/probe")
async def integration_probe(integration_id: str, request: IntegrationProbeRequest) -> dict[str, object]:
    try:
        return await probe_integration(request.user_id, integration_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"integration provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/github/repositories")
async def github_repositories(request: GithubRepositoriesRequest) -> dict[str, object]:
    try:
        return await list_github_repositories(request.user_id, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"github provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/github/issues")
async def github_issues(request: GithubIssuesRequest) -> dict[str, object]:
    try:
        return await list_github_issues(request.user_id, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"github provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/github/pull-requests")
async def github_pull_requests(request: GithubPullRequestsRequest) -> dict[str, object]:
    try:
        return await list_github_pull_requests(request.user_id, request.repository, request.page, request.per_page)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"github provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/github/create-issue")
async def github_create_issue_endpoint(request: GithubCreateIssueRequest) -> dict[str, object]:
    try:
        return await github_create_issue(request.user_id, request.repository, request.title, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"github provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/github/comment")
async def github_comment_endpoint(request: GithubCommentRequest) -> dict[str, object]:
    try:
        return await github_comment_issue_or_pr(request.user_id, request.repository, request.number, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"github provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/gmail/messages")
async def gmail_messages(request: GmailMessagesRequest) -> dict[str, object]:
    try:
        return await list_gmail_messages(request.user_id, request.query, request.page_token, request.max_results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/gmail/message")
async def gmail_message(request: GmailMessageRequest) -> dict[str, object]:
    try:
        return await get_gmail_message(request.user_id, request.message_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/gmail/send")
async def gmail_send(request: GmailSendRequest) -> dict[str, object]:
    try:
        return await send_gmail_message(request.user_id, request.to, request.subject, request.body, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"gmail provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
