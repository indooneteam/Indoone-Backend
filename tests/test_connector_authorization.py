import pytest

from app.ai.approval import decide_tool
from app.ai.connector_authorization import authorize_connector, authorize_tool, scope_allows
from app.capabilities.connector_registry import allows_capability
from app.capabilities.store import upsert_integration_token


def test_connector_registry_allows_registered_capability() -> None:
    assert allows_capability("gmail", "messages.read") is True
    assert allows_capability("gmail", "calendar.read") is False


def test_gmail_scope_aliases_cover_normalized_capabilities() -> None:
    assert scope_allows("gmail", "gmail.readonly", "messages.search") is True
    assert scope_allows("gmail", "gmail.readonly", "messages.read") is True
    assert scope_allows("gmail", "gmail.readonly", "messages.send") is False


def test_gmail_send_oauth_scope_allows_only_send() -> None:
    scope = "https://www.googleapis.com/auth/gmail.send"
    assert scope_allows("gmail", scope, "messages.send") is True
    assert scope_allows("gmail", scope, "messages.read") is False
    assert scope_allows("gmail", scope, "messages.search") is False


def test_connector_authorization_denies_missing_connection(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    result = authorize_connector("user-one", "gmail", "messages.read")
    assert result.allowed is False
    assert result.reason == "connector is not connected for this user"


def test_connector_authorization_denies_wrong_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "gmail", b"encrypted", None, "Bearer", "gmail.readonly", None)
    result = authorize_connector("user-one", "gmail", "messages.send")
    assert result.allowed is False
    assert result.reason == "connected token scope does not allow this capability"


def test_connector_authorization_denies_expired_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "gmail", b"encrypted", None, "Bearer", "gmail.readonly", "2000-01-01T00:00:00+00:00")
    result = authorize_connector("user-one", "gmail", "messages.read")
    assert result.allowed is False
    assert result.reason == "connector authorization has expired"


def test_connector_authorization_denies_malformed_expiry(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "gmail", b"encrypted", None, "Bearer", "gmail.readonly", "not-a-date")
    result = authorize_connector("user-one", "gmail", "messages.read")
    assert result.allowed is False
    assert result.reason == "connector authorization has expired"


@pytest.mark.parametrize("scope", ["gmail.modify", "gmail.readonly messages.read"])
def test_connector_authorization_allows_read_scope(monkeypatch, tmp_path, scope: str) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    upsert_integration_token("user-one", "gmail", b"encrypted", None, "Bearer", scope, None)
    result = authorize_tool("user-one", "gmail_read")
    assert result.allowed is True


def test_tool_decision_blocks_unconnected_connector_before_execution(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    decision = decide_tool("gmail_search", user_id="user-one")
    assert decision.allowed is False
    assert decision.reason == "connector is not connected for this user"


def test_tool_decision_still_allows_local_tool() -> None:
    decision = decide_tool("calculator", user_id="user-one")
    assert decision.allowed is True



@pytest.mark.parametrize("scope", ["public_repo", "read:user public_repo", "repo"])
def test_github_read_tools_are_granted_only_read_aliases(scope: str) -> None:
    from app.ai.connector_authorization import scope_allows

    for capability in (
        "repos.read", "repo.contents.read", "repo.files.read",
        "repo.code.search", "repo.history.read", "repo.compare.read",
    ):
        assert scope_allows("github", scope, capability) is True


def test_github_read_scope_does_not_grant_write_capabilities() -> None:
    from app.ai.connector_authorization import scope_allows

    assert scope_allows("github", "public_repo", "issues.write") is False
    assert scope_allows("github", "public_repo", "comments.write") is False
