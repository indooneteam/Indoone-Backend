"""Do not merge a pull request whose checks have not passed.

GitHub already refuses a merge on a protected branch when required checks are failing,
and the merge call already turns a 405 into "it may have conflicts or checks that have
not passed". That is honest but useless: it does not say *which* check, or whether the
work is still running, and it says it only after the attempt.

This module asks first. Before the merge call, it reads the pull request's head commit
and that commit's check runs, and returns what is blocking - named, and in the words a
person can act on:

    "The 'test' check failed."
    "2 checks are still running: test, postgres-store."
    "The checks could not be read, so nothing was merged."

The last one is the important one. A check that cannot be read is not a pass: an expired
token or a provider hiccup must not become a merge, which is the one GitHub action that
cannot be undone from the app.

What this deliberately does not do: refuse a merge when a repository has no checks at
all. A repository without CI is not a repository with failing CI, and blocking every
merge on it would make the assistant useless for small projects. There the merge is
left to GitHub's own protection rules.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

from app.plugins.github.service import (
    _GITHUB_REPOS_BASE_URL,
    _github_headers,
    _github_token,
    _validated_repository,
    list_github_check_runs,
)

logger = logging.getLogger("indoone.github.checks_gate")

# Conclusions that mean the work is not good to merge.
_BAD_CONCLUSIONS = {"failure", "timed_out", "cancelled", "action_required", "startup_failure"}
# Statuses that mean the work has not finished yet.
_PENDING_STATUSES = {"queued", "in_progress", "waiting", "requested", "pending"}


async def pull_request_head_sha(user_id: str, repository: str, number: int) -> str:
    """The head commit of a pull request, or an empty string when it cannot be read."""

    repo = _validated_repository(repository)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                f"{_GITHUB_REPOS_BASE_URL}/{repo}/pulls/{int(number)}",
                headers=_github_headers(_github_token(user_id.strip())),
            )
            response.raise_for_status()
            body = response.json()
    except Exception as exc:  # noqa: BLE001 - the caller decides what an unknown means
        logger.warning("could not read the pull request head: %s", exc)
        return ""
    if not isinstance(body, dict):
        return ""
    head = body.get("head")
    if not isinstance(head, dict):
        return ""
    return str(head.get("sha") or "").strip()


async def merge_blockers(user_id: str, repository: str, number: int) -> list[str]:
    """Everything standing between this pull request and a merge, in plain words.

    An empty list means nothing is blocking: the checks passed, or the repository has
    none. A non-empty list is a refusal.
    """

    user = str(user_id or "").strip()
    repo = str(repository or "").strip()
    if not user or not repo:
        return ["The repository could not be determined, so nothing was merged."]

    sha = await pull_request_head_sha(user, repo, number)
    if not sha:
        return [
            "The pull request's head commit could not be read, so nothing was merged. "
            "Check the pull request number and the GitHub connection, then try again."
        ]

    try:
        status = await list_github_check_runs(user, repo, sha)
    except Exception as exc:  # noqa: BLE001 - unreadable is not a pass
        logger.warning("could not read check runs for %s: %s", sha[:10], exc)
        return [
            "The checks for this pull request could not be read, so nothing was merged."
        ]

    runs = status.get("check_runs")
    runs = [run for run in runs if isinstance(run, dict)] if isinstance(runs, list) else []
    if not runs:
        # No CI configured for this commit. GitHub's own protection rules decide.
        return []

    failed = [
        str(run.get("name") or "a check")
        for run in runs
        if str(run.get("conclusion") or "") in _BAD_CONCLUSIONS
    ]
    pending = [
        str(run.get("name") or "a check")
        for run in runs
        if str(run.get("status") or "") in _PENDING_STATUSES
        and str(run.get("conclusion") or "") not in _BAD_CONCLUSIONS
    ]

    blockers: list[str] = []
    if failed:
        blockers.append(
            "These checks did not pass: " + ", ".join(sorted(set(failed))) + "."
        )
    if pending:
        blockers.append(
            "These checks are still running: " + ", ".join(sorted(set(pending))) + "."
        )
    return blockers
