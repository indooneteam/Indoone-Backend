import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.capabilities.github_writes import github_comment_issue_or_pr, github_create_issue
from app.capabilities.store import upsert_integration_token
from app.main import app


@pytest.fixture
def oauth_env(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("INDOONE_OAUTH_ENCRYPTION_KEY", key)
    cipher = Fernet(key.encode("ascii"))
    upsert_integration_token("user-one", "github", cipher.encrypt(b"user-one-secret"), None, "Bearer", "repo:write", None)
    return cipher


@pytest.mark.asyncio
async def test_github_create_issue_requires_explicit_approval(oauth_env):
    with pytest.raises(PermissionError, match="explicit approval"):
        await github_create_issue("user-one", "owner/repo", "hello", approved=False)


@pytest.mark.asyncio
async def test_github_create_issue_uses_user_token_and_fixed_endpoint(monkeypatch, oauth_env):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": 10, "number": 4, "html_url": "https://github.com/owner/repo/issues/4", "title": "hello", "state": "open"}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            assert url == "https://api.github.com/repos/owner/repo/issues"
            assert kwargs["headers"]["Authorization"] == "Bearer user-one-secret"
            assert kwargs["json"]["title"] == "hello"
            return Response()

    monkeypatch.setattr("app.capabilities.github_writes.httpx.AsyncClient", Client)
    result = await github_create_issue("user-one", "owner/repo", "hello", approved=True)
    assert result["operation"] == "create_issue"
    assert result["secrets_exposed"] is False


@pytest.mark.asyncio
async def test_github_comment_uses_approval_and_bounded_number(monkeypatch, oauth_env):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"id": 11, "html_url": "https://github.com/owner/repo/issues/4#issuecomment-11"}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            assert url == "https://api.github.com/repos/owner/repo/issues/4/comments"
            assert kwargs["headers"]["Authorization"] == "Bearer user-one-secret"
            return Response()

    monkeypatch.setattr("app.capabilities.github_writes.httpx.AsyncClient", Client)
    result = await github_comment_issue_or_pr("user-one", "owner/repo", 4, "approved comment", approved=True)
    assert result["operation"] == "comment"

    with pytest.raises(ValueError, match="number"):
        await github_comment_issue_or_pr("user-one", "owner/repo", 0, "x", approved=True)


def test_write_endpoints_reject_missing_approval(monkeypatch, tmp_path):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        issue = client.post("/api/integrations/github/create-issue", json={"user_id": "user-one", "repository": "owner/repo", "title": "hello"})
        comment = client.post("/api/integrations/github/comment", json={"user_id": "user-one", "repository": "owner/repo", "number": 1, "body": "hello"})
    assert issue.status_code == 403
    assert comment.status_code == 403
