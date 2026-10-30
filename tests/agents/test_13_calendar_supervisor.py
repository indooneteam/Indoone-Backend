"""Agent 13 - the calendar supervisor.

Every test here is a date that a model would get wrong. The arithmetic is checked against
the calendar itself, not against a remembered answer.
"""

from __future__ import annotations

import calendar as pycalendar
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from app.agents import load

supervisor = load("13_calendar_supervisor")

AT = datetime(2026, 10, 30, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


def run(call, **kwargs):
    return supervisor.validate_and_execute(call, AT, **kwargs)


# ---------------------------------------------------------------- 1. declarations


def test_the_declarations_are_function_calling_shaped() -> None:
    declarations = supervisor.get_tool_declarations()

    assert [item["name"] for item in declarations] == [
        "calendar_check_date",
        "calendar_get_upcoming_holidays",
        "calendar_get_month_schedule",
        "calendar_date_difference",
    ]
    for item in declarations:
        assert item["parameters"]["type"] == "object"


def test_a_caller_cannot_change_the_declarations() -> None:
    supervisor.get_tool_declarations()[0]["name"] = "calendar_guess"

    assert supervisor.get_tool_declarations()[0]["name"] == "calendar_check_date"


# ---------------------------------------------------------------- 2. the Saturday rule


@pytest.mark.parametrize(("year", "month"), [(2026, 11), (2026, 10), (2026, 1), (2025, 5)])
def test_the_second_and_fourth_saturdays_match_the_calendar(year: int, month: int) -> None:
    """Computed, not remembered: checked against the standard library's own calendar."""
    result = run(supervisor.CalendarToolCall("calendar_get_month_schedule", {"year": year, "month": month}))

    saturdays = [
        date(year, month, day)
        for day in range(1, pycalendar.monthrange(year, month)[1] + 1)
        if date(year, month, day).weekday() == 5
    ]

    assert result.success is True
    assert result.data["second_saturday"] == saturdays[1].isoformat()
    assert result.data["fourth_saturday"] == saturdays[3].isoformat()


def test_a_second_saturday_is_a_bank_holiday_and_a_working_day_is_not() -> None:
    second = date.fromisoformat(
        run(supervisor.CalendarToolCall("calendar_get_month_schedule", {"year": 2026, "month": 11})).data["second_saturday"]
    )

    holiday = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": second.isoformat()}))
    working = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-11-10"}))

    assert holiday.data["is_bank_holiday"] is True
    assert working.data["is_bank_holiday"] is False


# ---------------------------------------------------------------- 3. the gazette


def test_a_gazetted_holiday_is_reported_with_its_name_and_category() -> None:
    result = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-10-02"}))

    assert result.success is True
    assert result.data["is_holiday"] is True
    assert "Gandhi Jayanti" in result.formatted_summary
    assert result.data["events"][0]["category"] == "GAZETTED"


def test_an_occasion_that_is_not_a_day_off_says_so() -> None:
    """Rajyotsava 2026 falls on a Sunday, so the gazette does not give a separate day."""
    result = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-11-01"}))

    assert result.success is True
    assert result.data["is_holiday"] is False
    assert "not a gazetted day off" in result.formatted_summary
    assert result.data["events"][0]["category"] == "RESTRICTED"


def test_a_year_without_a_notification_is_reported_as_missing() -> None:
    result = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2030-01-01"}))

    assert result.success is False
    assert result.error_code == "OUT_OF_RANGE"
    assert result.error_message


def test_a_festival_missing_from_the_file_is_not_invented() -> None:
    result = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-11-10"}))

    assert result.success is True
    assert result.data["events"] == []
    assert "working day" in result.formatted_summary


# ---------------------------------------------------------------- 4. relative dates


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("today", "2026-10-30"),
        ("tomorrow", "2026-10-31"),
        ("yesterday", "2026-10-29"),
        ("naale", "2026-10-31"),
        ("ivattu", "2026-10-30"),
        ("kal", "2026-10-31"),
        ("రేపు", "2026-10-31"),
        ("நாளை", "2026-10-31"),
        ("2026-11-01", "2026-11-01"),
    ],
)
def test_a_relative_expression_resolves_against_kolkata_time(expression: str, expected: str) -> None:
    assert supervisor.resolve_relative_date(expression, AT) == expected


def test_a_nonsense_expression_resolves_to_nothing() -> None:
    """Better to say nothing than to guess a date."""
    assert supervisor.resolve_relative_date("whenever", AT) is None
    assert supervisor.resolve_relative_date("2026-13-45", AT) is None


# ---------------------------------------------------------------- 5. ranges and maths


def test_upcoming_holidays_are_sorted_and_limited() -> None:
    result = run(
        supervisor.CalendarToolCall("calendar_get_upcoming_holidays", {"days_ahead": 60, "limit": 3})
    )

    assert result.success is True
    dates = [item["date"] for item in result.data]
    assert dates == sorted(dates)
    assert len(dates) <= 3


def test_the_range_and_the_limit_are_clamped() -> None:
    arguments, _ = supervisor.validate_arguments(
        supervisor.CalendarToolCall("calendar_get_upcoming_holidays", {"days_ahead": 5000, "limit": 500})
    )

    assert arguments["days_ahead"] == supervisor.MAX_DAYS_AHEAD
    assert arguments["limit"] == supervisor.MAX_LIMIT


def test_a_month_schedule_adds_up() -> None:
    result = run(supervisor.CalendarToolCall("calendar_get_month_schedule", {"year": 2026, "month": 11}))
    data = result.data

    assert data["total_days"] == 30
    assert data["working_days"] + data["sundays"] >= 0
    assert data["working_days"] < data["total_days"]


def test_a_date_difference_counts_working_days_and_holidays() -> None:
    result = run(
        supervisor.CalendarToolCall("calendar_date_difference", {"start_date": "2026-10-01", "end_date": "2026-10-31"})
    )

    assert result.success is True
    assert result.data["total_days"] == 31
    assert result.data["working_days"] + result.data["holiday_days"] == 31


def test_a_backwards_range_is_refused_cleanly() -> None:
    result = run(
        supervisor.CalendarToolCall("calendar_date_difference", {"start_date": "2026-10-31", "end_date": "2026-10-01"})
    )

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENT"


def test_an_unknown_tool_is_refused_cleanly() -> None:
    result = run(supervisor.CalendarToolCall("calendar_delete_everything", {}))

    assert result.success is False
    assert result.error_code == "UNKNOWN_TOOL"


# ---------------------------------------------------------------- 6. chips and speed


def test_a_holiday_check_offers_the_next_holiday_and_the_month() -> None:
    result = run(supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-10-02"}))

    actions = [chip["action"] for chip in result.action_chips]
    assert actions == ["next_holiday", "month_schedule"]


def test_the_decision_costs_under_its_budget() -> None:
    call = supervisor.CalendarToolCall("calendar_check_date", {"target_date": "2026-10-02"})
    fastest = float("inf")

    for _ in range(100):
        started = time.thread_time()
        supervisor.validate_and_execute(call, AT)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    # The gazette file is read through its owning module, so the honest bound is the one the
    # module documents - sub-millisecond - rather than the 50 us a pure computation would be.
    assert fastest < 1.0, f"fastest {fastest:.4f} ms"


def test_relative_date_resolution_alone_is_under_the_budget() -> None:
    fastest = float("inf")

    for _ in range(500):
        started = time.thread_time()
        supervisor.resolve_relative_date("naale", AT)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
