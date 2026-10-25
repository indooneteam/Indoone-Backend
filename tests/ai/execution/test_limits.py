"""Admission control: the gate in front of the one endpoint that spends money.

Two layers are tested here because they fail differently. The burst limiter is in
process and protects the process; the daily quota is in the store and protects the
budget. A test that only covered one would leave the other free to regress.
"""

from __future__ import annotations

import pytest

from app.ai.execution import limits


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "limits.db"))
    monkeypatch.delenv("INDOONE_CHAT_LIMITS_ENABLED", raising=False)
    monkeypatch.delenv("INDOONE_CHAT_RATE_PER_MINUTE", raising=False)
    monkeypatch.delenv("INDOONE_CHAT_RATE_BURST", raising=False)
    monkeypatch.delenv("INDOONE_CHAT_DAILY_QUOTA", raising=False)
    limits.reset_burst_state()
    limits.initialize()
    yield
    limits.reset_burst_state()


# ---- the burst limiter ----------------------------------------------------- #


def test_a_burst_of_requests_is_refused_and_says_when_to_retry(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "3")
    monkeypatch.setenv("INDOONE_CHAT_RATE_PER_MINUTE", "30")

    allowed = [limits.admit("user-burst").allowed for _ in range(3)]
    refused = limits.admit("user-burst")

    assert allowed == [True, True, True]
    assert refused.allowed is False
    assert refused.code == limits.RATE_LIMITED
    assert refused.retry_after >= 1


def test_one_user_bursting_does_not_affect_another(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "2")

    for _ in range(2):
        limits.admit("user-a")
    assert limits.admit("user-a").allowed is False
    assert limits.admit("user-b").allowed is True


def test_the_bucket_refills_over_time(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "1")
    monkeypatch.setenv("INDOONE_CHAT_RATE_PER_MINUTE", "60")

    now = 1_000.0
    assert limits._take_burst("user-refill", now=now).allowed is True
    assert limits._take_burst("user-refill", now=now).allowed is False
    # One token per second at 60 a minute, so a second later there is one again.
    assert limits._take_burst("user-refill", now=now + 1.0).allowed is True


def test_a_refused_burst_does_not_spend_daily_quota(monkeypatch) -> None:
    """A client retrying too fast must not also lose its day's allowance."""

    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "1")

    limits.admit("user-charge")
    assert limits.admit("user-charge").allowed is False

    assert limits.usage_today("user-charge") == 1


def test_an_unauthenticated_caller_is_keyed_by_address(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "1")

    assert limits.admit("", client_ip="203.0.113.9").allowed is True
    assert limits.admit("", client_ip="203.0.113.9").allowed is False
    assert limits.admit("", client_ip="203.0.113.10").allowed is True


# ---- the daily quota ------------------------------------------------------- #


def test_the_daily_quota_refuses_past_the_limit(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_DAILY_QUOTA", "3")
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "100")

    results = [limits.admit("user-quota") for _ in range(4)]

    assert [item.allowed for item in results] == [True, True, True, False]
    assert results[-1].code == limits.DAILY_QUOTA_EXCEEDED
    assert results[-1].retry_after >= 1
    assert results[0].remaining_today == 2


def test_the_counter_survives_a_restart_of_the_process_state(monkeypatch) -> None:
    """The quota is in the store, so clearing the in-process buckets must not reset it."""

    monkeypatch.setenv("INDOONE_CHAT_DAILY_QUOTA", "2")
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "100")

    limits.admit("user-restart")
    limits.admit("user-restart")
    limits.reset_burst_state()

    assert limits.usage_today("user-restart") == 2
    assert limits.admit("user-restart").allowed is False


def test_a_new_day_starts_from_zero() -> None:
    limits._consume_daily("user-window", window="2026-10-24")
    limits._consume_daily("user-window", window="2026-10-24")

    assert limits.usage_today("user-window", window="2026-10-24") == 2
    assert limits.usage_today("user-window", window="2026-10-25") == 0


# ---- the failure modes ----------------------------------------------------- #


def test_limits_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CHAT_LIMITS_ENABLED", "0")
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "1")

    assert limits.admit("user-off").allowed is True
    assert limits.admit("user-off").allowed is True
    assert limits.usage_today("user-off") == 0


def test_a_store_that_cannot_answer_does_not_lock_users_out(monkeypatch) -> None:
    """The burst limiter has already run, so a broken store degrades rather than blocks."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(limits.database, "connect", _boom)
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "100")

    assert limits.admit("user-broken-store").allowed is True
    assert limits.usage_today("user-broken-store") == 0


def test_a_bad_setting_falls_back_to_the_default(monkeypatch) -> None:
    """A setting that cannot be read must not become a limit of zero.

    "not-a-number" and "-5" are both operator mistakes. The safe reading is the
    default, not a value that refuses every request.
    """

    monkeypatch.setenv("INDOONE_CHAT_DAILY_QUOTA", "not-a-number")
    monkeypatch.setenv("INDOONE_CHAT_RATE_BURST", "-5")

    assert limits.daily_quota() == limits._DEFAULT_DAILY_QUOTA
    assert limits._burst() == limits._DEFAULT_BURST
    assert limits._burst() >= 1
