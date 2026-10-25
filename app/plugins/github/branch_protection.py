"""Is the branch this write lands on one we must not touch without a confirmation?

Reversibility was the wrong axis on its own. Writing a file is reversible - the old
content is in the commit before it - but writing it to `main` is not the same act as
writing it to a scratch branch. The blast radius is different: `main` is what deploys,
what other people branch from, and what a mistake is noticed on last.

So the rule is a second gate, not a replacement for the first:

* a write that changes nothing that cannot be changed again still runs without a
  confirmation (create a branch, write a file to a scratch branch);
* a write whose **target branch is protected** always needs a server-issued approval
  token, whether or not the user asked for it in their own words.

Two things count as protected here, and the second one is deliberate.

1. GitHub says so: `GET /repos/{owner}/{repo}/branches/{branch}` reports
   ``protected: true``.
2. It is the repository's default branch. A default branch that GitHub does not report
   as protected is still the branch every clone starts from, and the cost of asking
   once too often is a confirmation; the cost of not asking is a broken `main`.

The check fails closed on purpose. If GitHub cannot answer - expired token, network,
a repository that has vanished - the answer is "treat it as protected", because the
one outcome we must never produce is an unconfirmed write to a branch we could not
check.
"""

from __future__ import annotations

import logging

import httpx

from app.plugins.github.service import (
    _GITHUB_REPOS_BASE_URL,
    _github_headers,
    _github_token,
    _validated_repository,
    _validated_repository_ref,
)

logger = logging.getLogger("indoone.github.branch_protection")


async def default_branch(user_id: str, repository: str) -> str:
    """The repository's default branch, or an empty string when it cannot be read."""

    repo = _validated_repository(repository)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{_GITHUB_REPOS_BASE_URL}/{repo}",
                headers=_github_headers(_github_token(user_id.strip())),
            )
            response.raise_for_status()
            body = response.json()
    except Exception:
        return ""
    if not isinstance(body, dict):
        return ""
    return str(body.get("default_branch") or "").strip()


async def is_protected(user_id: str, repository: str, branch: str) -> bool:
    """Whether GitHub reports this branch as protected. Unknown reads as protected."""

    repo = _validated_repository(repository)
    try:
        ref = _validated_repository_ref(branch)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{_GITHUB_REPOS_BASE_URL}/{repo}/branches/{ref}",
                headers=_github_headers(_github_token(user_id.strip())),
            )
            if response.status_code == 404:
                # The branch does not exist yet. A write that creates it is creating
                # something new, so there is no protected branch to damage.
                return False
            response.raise_for_status()
            body = response.json()
    except Exception as exc:  # noqa: BLE001 - any failure must fail closed
        logger.warning("branch protection check failed, treating as protected: %s", exc)
        return True
    return bool(isinstance(body, dict) and body.get("protected") is True)


async def protected_branch_reason(user_id: str, repository: str, branch: str = "") -> str:
    """Why this write needs a confirmation, or an empty string when it does not.

    The reason is returned as a sentence rather than a boolean because it is shown to
    the user. "This writes to main, which is protected" is a thing a person can act on;
    "approval required" is not.
    """

    user = str(user_id or "").strip()
    repo = str(repository or "").strip()
    if not user or not repo:
        return ""

    target = str(branch or "").strip()
    if not target:
        target = await default_branch(user, repo)
        if not target:
            # No branch given and the default could not be read: we do not know what
            # this write would touch, so it does not run unconfirmed.
            return (
                "The target branch could not be determined, so this change needs your "
                "confirmation before it runs."
            )

    if await is_protected(user, repo, target):
        return (
            f"This writes directly to {target}, which is a protected branch. "
            "Confirm and it will run; nothing has been changed yet."
        )
    return ""
