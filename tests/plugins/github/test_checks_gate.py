"""A merge waits for the checks, and says which one is in the way.

GitHub already refuses a merge on a protected branch when required checks fail, and the
merge call already turns a 405 into "it may have conflicts or checks that have not
passed". That sentence names nothing. These tests hold the gate that asks first.
"""

from __future__ import annotations

import asyncio

import pytest

from app.plugins.github import checks_gate


def _run(coro):
    return asyncio.run(coro)


def _patch(monkeypatch, *, sha: str = "a" * 40, runs: list[dict] | None = None, raises: Exception | None = None):
    async def fake_head(*_args, **_kwargs):
        return sha

    async def fake_checks(*_args, **_kwargs):
        if raises is not None:
            raise raises
        return {"check_runs": runs or [], "conclusion": "x"}

    monkeypatch.setattr(checks_gate, "pull_request_head_sha", fake_head)
    monkeypatch.setattr(checks_gate, "list_github_check_runs", fake_checks)


def test_a_green_pull_request_is_not_blocked(monkeypatch) -> None:
    _patch(monkeypatch, runs=[{"name": "test", "status": "completed", "conclusion": "success"}])

    assert _run(checks_gate.merge_blockers("u1", "o/r", 7)) == []


def test_a_failed_check_is_named(monkeypatch) -> None:
    _patch(
        monkeypatch,
        runs=[
            {"name": "test", "status": "completed", "conclusion": "failure"},
            {"name": "postgres-store", "status": "completed", "conclusion": "success"},
        ],
    )

    blockers = _run(checks_gate.merge_blockers("u1", "o/r", 7))

    assert len(blockers) == 1
    assert "test" in blockers[0]
    assert "did not pass" in blockers[0]
    # The one that passed must not be blamed.
    assert "postgres-store" not in blockers[0]


def test_a_running_check_is_named_as_still_running(monkeypatch) -> None:
    _patch(
        monkeypatch,
        runs=[
            {"name": "test", "status": "in_progress", "conclusion": ""},
            {"name": "postgres-store", "status": "queued", "conclusion": ""},
        ],
    )

    blockers = _run(checks_gate.merge_blockers("u1", "o/r", 7))

    assert len(blockers) == 1
    assert "still running" in blockers[0]
    assert "test" in blockers[0] and "postgres-store" in blockers[0]


def test_a_failure_and_a_running_check_are_reported_separately(monkeypatch) -> None:
    _patch(
        monkeypatch,
        runs=[
            {"name": "test", "status": "completed", "conclusion": "failure"},
            {"name": "lint", "status": "in_progress", "conclusion": ""},
        ],
    )

    blockers = _run(checks_gate.merge_blockers("u1", "o/r", 7))

    assert len(blockers) == 2


def test_a_repository_with_no_checks_is_not_blocked(monkeypatch) -> None:
    """No CI is not failing CI. Small repositories must still be mergeable."""

    _patch(monkeypatch, runs=[])

    assert _run(checks_gate.merge_blockers("u1", "o/r", 7)) == []


def test_checks_that_cannot_be_read_block_the_merge(monkeypatch) -> None:
    """An expired token must not become a merge - it is the one action we cannot undo."""

    _patch(monkeypatch, raises=RuntimeError("401 unauthorized"))

    blockers = _run(checks_gate.merge_blockers("u1", "o/r", 7))

    assert blockers and "could not be read" in blockers[0]


def test_an_unreadable_head_blocks_the_merge(monkeypatch) -> None:
    _patch(monkeypatch, sha="")

    blockers = _run(checks_gate.merge_blockers("u1", "o/r", 7))

    assert blockers and "head commit could not be read" in blockers[0]


def test_an_empty_repository_blocks_the_merge(monkeypatch) -> None:
    _patch(monkeypatch)

    assert _run(checks_gate.merge_blockers("u1", "", 7))


@pytest.mark.parametrize("conclusion", ["timed_out", "cancelled", "action_required", "startup_failure"])
def test_every_bad_conclusion_counts_as_a_failure(monkeypatch, conclusion: str) -> None:
    _patch(monkeypatch, runs=[{"name": "test", "status": "completed", "conclusion": conclusion}])

    assert _run(checks_gate.merge_blockers("u1", "o/r", 7))


def test_the_merge_call_refuses_before_it_asks_github_to_merge(monkeypatch) -> None:
    """The gate runs before the PUT, so a blocked merge never reaches the API."""

    from app.plugins.github import writes

    async def blocking(*_args, **_kwargs):
        return ["These checks did not pass: test."]

    calls: list[str] = []

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def put(self, url, **_kwargs):
            calls.append(url)
            raise AssertionError("the merge must not be attempted")

    monkeypatch.setattr("app.plugins.github.checks_gate.merge_blockers", blocking)
    monkeypatch.setattr(writes.httpx, "AsyncClient", _Client)

    with pytest.raises(RuntimeError, match="did not pass"):
        _run(
            writes.github_merge_pull_request(
                user_id="u1", repository="o/r", number=7, approved=True
            )
        )

    assert calls == []
