undefined


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
