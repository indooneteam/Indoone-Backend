"""Agent 22 - the quota guard.

The ceiling is the point: Meta's free tier is a thousand conversations, and an invoice is a
worse way to find out you crossed it than a redirect is.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agents import load

guard = load("22_omnichannel_quota_guard")

KOLKATA = ZoneInfo("Asia/Kolkata")
OCT_30 = datetime(2026, 10, 30, 10, 0, tzinfo=KOLKATA)
OCT_31_LATE = datetime(2026, 10, 31, 23, 59, tzinfo=KOLKATA)
NOV_1 = datetime(2026, 11, 1, 0, 0, tzinfo=KOLKATA)


@pytest.fixture(autouse=True)
def _fresh():
    guard.reset_state_for_tests()
    yield
    guard.reset_state_for_tests()


# ---------------------------------------------------------------- 1. the ceiling


def test_the_first_nine_hundred_conversations_are_allowed() -> None:
    for index in range(899):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    result = guard.check_quota("wa:919999999999", OCT_30)

    assert result.is_allowed is True
    assert result.current_month_usage == 899
    assert result.monthly_limit == guard.WHATSAPP_MONTHLY_LIMIT


def test_the_nine_hundred_and_first_is_blocked() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    result = guard.check_quota("wa:919999999999", OCT_30)

    assert result.is_allowed is False
    assert result.error_code == "WA_MONTHLY_QUOTA_EXHAUSTED"
    assert result.quota_exhausted_message


def test_the_block_is_a_redirect_to_the_app() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    result = guard.check_quota("wa:919999999999", OCT_30)

    assert [chip["action"] for chip in result.action_chips] == ["open_app_download"]


def test_the_message_is_localised() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    kannada = guard.check_quota("wa:919999999999", OCT_30, language="kn")
    hindi = guard.check_quota("wa:919999999999", OCT_30, language="hi")
    english = guard.check_quota("wa:919999999999", OCT_30, language="en")

    assert kannada.quota_exhausted_message != english.quota_exhausted_message
    assert hindi.quota_exhausted_message != english.quota_exhausted_message


def test_one_conversation_counts_once_however_many_turns() -> None:
    """Meta bills a conversation, not a message."""
    for _ in range(50):
        guard.record_usage("wa:919876543210", OCT_30)

    assert guard.check_quota("wa:919876543210", OCT_30).current_month_usage == 1


# ---------------------------------------------------------------- 2. the rollover


def test_the_counter_resets_on_the_first_of_the_month() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_31_LATE)

    assert guard.check_quota("wa:919999999999", OCT_31_LATE).is_allowed is False

    first = guard.check_quota("wa:919999999999", NOV_1)

    assert first.is_allowed is True
    assert first.current_month_usage == 0


def test_a_naive_timestamp_is_read_as_kolkata_time() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", datetime(2026, 10, 30, 10, 0))

    assert guard.check_quota("wa:919999999999", datetime(2026, 10, 30, 11, 0)).is_allowed is False


# ---------------------------------------------------------------- 3. the open funnel


def test_telegram_and_instagram_are_never_capped() -> None:
    for index in range(900):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    for channel in ("tg:987654321", "ig:17841400123456789"):
        result = guard.check_quota(channel, OCT_30)

        assert result.is_allowed is True
        assert result.monthly_limit is None
        assert result.channel in {"telegram", "instagram"}


def test_a_thousand_turns_on_telegram_are_still_allowed() -> None:
    for index in range(1200):
        guard.record_usage(f"tg:{index}", OCT_30)

    assert guard.check_quota("tg:1201", OCT_30).is_allowed is True


def test_an_app_user_is_not_capped() -> None:
    result = guard.check_quota("user-1234", OCT_30)

    assert result.is_allowed is True
    assert result.channel == "app"


# ---------------------------------------------------------------- 4. the buffer


def test_only_the_last_five_turns_are_kept() -> None:
    for index in range(8):
        guard.store_turn("wa:1", f"q{index}", f"a{index}")

    history = guard.get_recent_history("wa:1")
    prompts = [text for role, text in history if role == "user"]

    assert prompts == ["q3", "q4", "q5", "q6", "q7"]


def test_the_history_is_in_order_and_paired() -> None:
    guard.store_turn("wa:1", "first", "one")
    guard.store_turn("wa:1", "second", "two")

    assert guard.get_recent_history("wa:1") == [
        ("user", "first"),
        ("assistant", "one"),
        ("user", "second"),
        ("assistant", "two"),
    ]


def test_a_turn_older_than_fifteen_minutes_is_dropped() -> None:
    now = time.time()
    guard.store_turn("wa:1", "old question", "old answer", now=now - 1000)
    guard.store_turn("wa:1", "fresh question", "fresh answer", now=now)

    history = guard.get_recent_history("wa:1", now=now)

    assert ("user", "old question") not in history
    assert ("user", "fresh question") in history


def test_a_session_with_nothing_fresh_is_forgotten() -> None:
    now = time.time()
    guard.store_turn("wa:1", "old", "old", now=now - 5000)

    assert guard.get_recent_history("wa:1", now=now) == []
    assert "wa:1" not in guard._sessions


def test_pruning_drops_the_quiet_sessions() -> None:
    now = time.time()
    guard.store_turn("wa:quiet", "q", "a", now=now - 5000)
    guard.store_turn("wa:busy", "q", "a", now=now)

    dropped = guard.prune(now=now)

    assert dropped == 1
    assert "wa:quiet" not in guard._sessions
    assert "wa:busy" in guard._sessions


def test_an_unknown_user_has_no_history() -> None:
    assert guard.get_recent_history("wa:nobody") == []


def test_the_session_map_stays_bounded() -> None:
    for index in range(guard.MAX_SESSIONS + 50):
        guard.store_turn(f"wa:{index}", "q", "a")

    assert len(guard._sessions) <= guard.MAX_SESSIONS


# ---------------------------------------------------------------- 5. speed


def test_a_quota_check_stays_under_its_budget() -> None:
    for index in range(500):
        guard.record_usage(f"wa:9190000{index:05d}", OCT_30)

    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        guard.check_quota("wa:919999999999", OCT_30)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_a_history_fetch_stays_under_its_budget() -> None:
    guard.store_turn("wa:1", "q", "a")
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        guard.get_recent_history("wa:1")
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
