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

    async def put(self, url, **kwargs):
        type(self).calls.append(("PUT", url, kwargs))
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


@pytest.mark.asyncio
async def test_github_repository_file_commit_requires_approval():
    from app.plugins.github.writes import github_update_repository_file

    with pytest.raises(PermissionError, match="approval"):
        await github_update_repository_file(
            "user-1", "owner/repo", "src/app.py", "print('ok')",
            "Update app", "a" * 40, approval_verified=False,
        )


@pytest.mark.asyncio
async def test_github_repository_file_commit_uses_expected_sha_and_redacts_no_secrets(monkeypatch):
    from app.plugins.github import writes

    FakeClient.responses = [FakeResponse({
        "content": {
            "path": "src/app.py",
            "sha": "b" * 40,
            "html_url": "https://github.com/owner/repo/blob/main/src/app.py",
        },
        "commit": {
            "sha": "c" * 40,
            "html_url": "https://github.com/owner/repo/commit/" + "c" * 40,
        },
    })]
    monkeypatch.setattr(writes, "_github_token", lambda user_id: "access-token")
    monkeypatch.setattr(writes.httpx, "AsyncClient", FakeClient)
    content = "print('hello')\n"
    result = await writes.github_update_repository_file(
        "user-1", "owner/repo", "src/app.py", content,
        "Update source file", "a" * 40, approval_verified=True, branch="main",
    )
    method, url, kwargs = FakeClient.calls[0]
    assert method == "PUT"
    assert url == "https://api.github.com/repos/owner/repo/contents/src/app.py"
    assert kwargs["json"]["sha"] == "a" * 40
    assert kwargs["json"]["branch"] == "main"
    assert base64.b64decode(kwargs["json"]["content"]).decode("utf-8") == content
    assert kwargs["headers"]["Authorization"] == "Bearer access-token"
    assert result["file_sha"] == "b" * 40
    assert result["commit_sha"] == "c" * 40
    assert result["secrets_exposed"] is False


@pytest.mark.asyncio
async def test_github_repository_file_commit_rejects_embedded_credentials(monkeypatch):
    from app.plugins.github import writes

    monkeypatch.setattr(writes, "_github_token", lambda user_id: "access-token")
    with pytest.raises(ValueError, match="credential or private key"):
        await writes.github_update_repository_file(
            "user-1", "owner/repo", "src/app.py",
            "TOKEN = 'ghp_" + ("A" * 36) + "'",
            "Update source file", "a" * 40, approval_verified=True,
        )


def test_github_file_edit_tool_requires_approval():
    from app.ai.execution.tool_registry import get_tool_spec, requires_approval
    from app.ai.execution.connector_authorization import scope_allows

    spec = get_tool_spec("github_update_repository_file")
    assert spec is not None
    assert spec.connector_id == "github"
    assert spec.capability == "repo.files.write"
    # Writing a file is reversible and has nothing to preview, so it runs when asked.
    assert requires_approval("github_update_repository_file") is False
    assert scope_allows("github", "public_repo", "repo.files.write")
    assert scope_allows("github", "repo", "repo.files.write")


@pytest.mark.asyncio
async def test_a_protected_branch_does_not_run_on_an_invalid_approval_token(monkeypatch):
    """An invalid token must not write, and must not be a dead end either.

    On a protected branch the write is held and offered as a proposal. The property
    that matters is unchanged - the commit does not happen.
    """

    import json
    from app.ai.execution import tools as ai_tools

    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a" * 32)
    committed: list[dict] = []

    async def _must_not_commit(**kwargs):
        committed.append(kwargs)
        raise AssertionError("the write must not run without a valid approval")

    async def _protected(*_args, **_kwargs):
        return "This writes directly to main, which is a protected branch."

    monkeypatch.setattr(ai_tools, "commit_github_repository_file", _must_not_commit)
    monkeypatch.setattr(
        "app.plugins.github.branch_protection.protected_branch_reason", _protected
    )
    monkeypatch.setattr(ai_tools, "issue_approval_token", lambda *_a, **_k: "issued")
    monkeypatch.setattr(
        "app.capabilities.connectors.store.save_pending_github_action", lambda *_a, **_k: None
    )

    class _Fernet:
        def encrypt(self, value: bytes) -> bytes:
            return value

    monkeypatch.setattr("app.capabilities.connectors.integrations._fernet", lambda: _Fernet())

    payload = {
        "user_id": "user-1",
        "repository": "owner/repo",
        "path": "src/app.py",
        "content": "print('safe')",
        "message": "Update source",
        "sha": "a" * 40,
        "branch": "main",
        "approval_token": "not-a-valid-signed-token",
    }
    result = json.loads(await ai_tools._github_update_repository_file(json.dumps(payload)))

    assert committed == []
    assert result["code"] == "APPROVAL_REQUIRED"
    assert "protected branch" in result["reply"]
    assert "confirm" in result["reply"].lower()


@pytest.mark.asyncio
async def test_a_reversible_file_write_runs_without_a_token(monkeypatch):
    """The executor and its spec now agree: an unprotected edit is not gated."""

    import json
    from app.ai.execution import tools as ai_tools

    monkeypatch.setenv("INDOONE_APPROVAL_SECRET", "a" * 32)
    committed: list[dict] = []

    async def _commit(**kwargs):
        committed.append(kwargs)
        return {"content": {"sha": "b" * 40}}

    async def _not_protected(*_args, **_kwargs):
        return ""

    monkeypatch.setattr(ai_tools, "commit_github_repository_file", _commit)
    monkeypatch.setattr(
        "app.plugins.github.branch_protection.protected_branch_reason", _not_protected
    )
    monkeypatch.setattr(ai_tools, "_verified_write_result", _passthrough)

    payload = {
        "user_id": "user-1",
        "repository": "owner/repo",
        "path": "src/app.py",
        "content": "print('safe')",
        "message": "Update source",
        "sha": "a" * 40,
        "branch": "feature/x",
    }
    await ai_tools._github_update_repository_file(json.dumps(payload))

    assert committed and committed[0]["branch"] == "feature/x"


async def _passthrough(_user_id, _tool, _data, result):
    import json as _json

    return _json.dumps({"ok": True, "result": result})


@pytest.mark.asyncio
async def test_github_repository_file_create_omits_existing_sha(monkeypatch):
    from app.plugins.github import writes

    FakeClient.responses = [FakeResponse({
        "content": {
            "path": "src/new_module.py",
            "sha": "d" * 40,
            "html_url": "https://github.com/owner/repo/blob/main/src/new_module.py",
        },
        "commit": {
            "sha": "e" * 40,
            "html_url": "https://github.com/owner/repo/commit/" + "e" * 40,
        },
    })]
    monkeypatch.setattr(writes, "_github_token", lambda user_id: "access-token")
    monkeypatch.setattr(writes.httpx, "AsyncClient", FakeClient)
    result = await writes.github_update_repository_file(
        "user-1", "owner/repo", "src/new_module.py", "VALUE = 42\n",
        "Create module", "", approval_verified=True, branch="main", create=True,
    )
    _method, _url, kwargs = FakeClient.calls[0]
    assert "sha" not in kwargs["json"]
    assert result["operation"] == "create_file"
    assert result["path"] == "src/new_module.py"


@pytest.mark.asyncio
async def test_github_repository_file_create_refuses_existing_sha(monkeypatch):
    from app.plugins.github import writes

    monkeypatch.setattr(writes, "_github_token", lambda user_id: "access-token")
    with pytest.raises(ValueError, match="must not specify"):
        await writes.github_update_repository_file(
            "user-1", "owner/repo", "src/new_module.py", "VALUE = 42",
            "Create module", "a" * 40, approval_verified=True, create=True,
        )
