import base64

import pytest

from app.plugins.github import service


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FakeClient:
    responses = []
    calls = []

    def __init__(self, *args, **kwargs):
        self.responses = type(self).responses
        type(self).calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url, **kwargs):
        type(self).calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_repository_contents_returns_only_safe_entry_fields(monkeypatch):
    FakeClient.responses = [FakeResponse([
        {"name": "app", "path": "app", "type": "dir", "size": 0, "html_url": "https://github.com/owner/repo/tree/main/app", "private_token": "do-not-return"},
        {"name": "README.md", "path": "README.md", "type": "file", "size": 42, "html_url": "https://github.com/owner/repo/blob/main/README.md"},
    ])]
    monkeypatch.setattr(service, "_github_token", lambda user_id: "access-token")
    monkeypatch.setattr(service.httpx, "AsyncClient", FakeClient)
    result = await service.list_github_repository_contents("user-1", "owner/repo", ref="main")
    assert result["count"] == 2
    assert result["contents"][0] == {
        "name": "app", "path": "app", "type": "dir", "size": 0,
        "html_url": "https://github.com/owner/repo/tree/main/app",
    }
    url, kwargs = FakeClient.calls[0]
    assert url == "https://api.github.com/repos/owner/repo/contents"
    assert kwargs["params"] == {"ref": "main"}
    assert kwargs["headers"]["Authorization"] == "Bearer access-token"
    assert result["secrets_exposed"] is False


@pytest.mark.asyncio
async def test_read_file_decodes_content_and_redacts_known_github_token(monkeypatch):
    content = "print('hello')\\nTOKEN = 'ghp_" + ("A" * 36) + "'\\n"
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    FakeClient.responses = [FakeResponse({
        "type": "file", "name": "sample.py", "path": "sample.py", "size": len(content),
        "sha": "abc123", "html_url": "https://github.com/owner/repo/blob/main/sample.py",
        "encoding": "base64", "content": encoded,
    })]
    monkeypatch.setattr(service, "_github_token", lambda user_id: "access-token")
    monkeypatch.setattr(service.httpx, "AsyncClient", FakeClient)
    result = await service.read_github_repository_file("user-1", "owner/repo", "sample.py")
    assert "print('hello')" in result["content"]
    assert "ghp_" not in result["content"]
    assert "[REDACTED_GITHUB_TOKEN]" in result["content"]


@pytest.mark.asyncio
async def test_repository_path_rejects_parent_traversal():
    with pytest.raises(ValueError, match="parent segments"):
        await service.list_github_repository_contents("user-1", "owner/repo", "../secrets")


@pytest.mark.asyncio
async def test_search_history_and_compare_are_read_only(monkeypatch):
    FakeClient.responses = [
        FakeResponse({"total_count": 1, "items": [{
            "name": "auth.py", "path": "app/auth.py", "html_url": "https://github.com/owner/repo/blob/main/app/auth.py",
            "text_matches": [{"fragment": "def authenticate(user): return user"}],
        }]}),
        FakeResponse([{"sha": "1234567890", "commit": {"message": "fix: auth", "author": {"name": "Dev", "date": "2026-01-01T00:00:00Z"}}, "html_url": "https://github.com/owner/repo/commit/123"}]),
        FakeResponse({"status": "ahead", "ahead_by": 1, "behind_by": 0, "total_commits": 1, "files": [{"filename": "app/auth.py", "status": "modified", "additions": 2, "deletions": 1, "changes": 3, "blob_url": "https://github.com/owner/repo/blob/main/app/auth.py"}]}),
    ]
    monkeypatch.setattr(service, "_github_token", lambda user_id: "access-token")
    monkeypatch.setattr(service.httpx, "AsyncClient", FakeClient)
    search = await service.search_github_repository_code("user-1", "owner/repo", "authenticate")
    history = await service.list_github_repository_commits("user-1", "owner/repo")
    compare = await service.compare_github_repository_commits("user-1", "owner/repo", "main", "develop")
    assert search["results"][0]["path"] == "app/auth.py"
    assert history["commits"][0]["message"] == "fix: auth"
    assert compare["ahead_by"] == 1
    assert compare["files"][0]["filename"] == "app/auth.py"
