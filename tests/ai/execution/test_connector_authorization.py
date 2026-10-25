import pytest

from app.ai.execution.approval import decide_tool
from app.ai.execution.connector_authorization import authorize_connector, authorize_tool, scope_allows
from app.capabilities.connectors.connector_registry import allows_capability
from app.capabilities.connectors.store import upsert_integration_token


def test_connector_registry_allows_registered_capability() -> None:
    assert allows_capability("github", "repos.read") is True
    assert allows_capability("github", "calendar.read") is False


def test_connector_authorization_denies_missing_connection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    result = authorize_connector("user-one", "github", "repos.read")
    assert result.allowed is False
    assert result.reason == "connector is not connected for this user"


def test_connector_authorization_denies_wrong_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "github", b"encrypted", None, "Bearer", "public_repo", None)
    # `issues.write` is registered on the connector but a public_repo token does not
    # grant it. The previous example here was `pull_requests.write`, which a
    # public_repo token does grant - the scope map was missing it, so the test was
    # asserting a bug as if it were the rule.
    result = authorize_connector("user-one", "github", "issues.write")
    assert result.allowed is False
    assert result.reason == "connected token scope does not allow this capability"


@pytest.mark.parametrize("capability", ["branches.write", "pull_requests.write"])
def test_connector_authorization_allows_branch_and_pull_request_writes(
    monkeypatch, tmp_path, capability: str
) -> None:
    """The scope map must grant what the token actually grants.

    github_create_branch, github_delete_branch, github_create_pull_request and
    github_merge_pull_request all declare these capabilities. A repo/public_repo
    token grants them, so the map must say so; when it did not, every branch and
    pull-request write was refused with "connected token scope does not allow this
    capability" before the approval step could even be reached.
    """

    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    for scope in ("repo", "public_repo"):
        upsert_integration_token("user-two", "github", b"encrypted", None, "Bearer", scope, None)
        result = authorize_connector("user-two", "github", capability)
        assert result.allowed is True, (scope, capability, result.reason)
        assert scope_allows("github", scope, capability) is True


def test_connector_authorization_denies_expired_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "github", b"encrypted", None, "Bearer", "public_repo", "2000-01-01T00:00:00+00:00")
    result = authorize_connector("user-one", "github", "repos.read")
    assert result.allowed is False
    assert result.reason == "connector authorization has expired"


def test_connector_authorization_denies_malformed_expiry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "github", b"encrypted", None, "Bearer", "public_repo", "not-a-date")
    result = authorize_connector("user-one", "github", "repos.read")
    assert result.allowed is False
    assert result.reason == "connector authorization has expired"


@pytest.mark.parametrize("scope", ["repo", "repo repos.read"])
def test_connector_authorization_allows_read_scope(monkeypatch, tmp_path, scope: str) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "github", b"encrypted", None, "Bearer", scope, None)
    result = authorize_tool("user-one", "github_read_repository_file")
    assert result.allowed is True


def test_tool_decision_blocks_unconnected_connector_before_execution(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    decision = decide_tool("github_list_repositories", user_id="user-one")
    assert decision.allowed is False
    assert decision.reason == "connector is not connected for this user"


def test_tool_decision_still_allows_local_tool() -> None:
    decision = decide_tool("calculator", user_id="user-one")
    assert decision.allowed is True



@pytest.mark.parametrize("scope", ["public_repo", "read:user public_repo", "repo"])
def test_github_read_tools_are_granted_only_read_aliases(scope: str) -> None:
    from app.ai.execution.connector_authorization import scope_allows

    for capability in (
        "repos.read", "repo.contents.read", "repo.files.read",
        "repo.code.search", "repo.history.read", "repo.compare.read",
    ):
        assert scope_allows("github", scope, capability) is True


def test_github_read_scope_does_not_grant_write_capabilities() -> None:
    from app.ai.execution.connector_authorization import scope_allows

    assert scope_allows("github", "public_repo", "issues.write") is False
    assert scope_allows("github", "public_repo", "comments.write") is False
