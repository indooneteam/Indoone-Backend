"""More than one year of the calendar, and an honest answer when there is no year.

The gazette is published in November or December of the preceding year, so the 2027 list
does not exist yet. The tests hold both halves of that: a covered year answers from its
file, and an uncovered year says so and gives only the dates that cannot move.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.plugins.festivals.calendar import (
    covered_years,
    festival_answer,
    resolve_year,
)

INDIA = ZoneInfo("Asia/Kolkata")


def at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 9, 0, tzinfo=INDIA)


def test_the_files_that_exist_are_the_years_that_answer() -> None:
    years = covered_years()

    assert 2025 in years and 2026 in years
    assert years == tuple(sorted(years))


def test_a_question_without_a_year_uses_the_current_one() -> None:
    assert resolve_year("when is deepavali", at(2025, 6, 1).date()) == 2025
    assert resolve_year("2027 ugadi date", at(2026, 6, 1).date()) == 2027


def test_a_2025_festival_answers_from_the_2025_file() -> None:
    answer = festival_answer("when is Ugadi", now=at(2025, 1, 1))

    assert answer is not None
    assert "2025-03-30" in answer
    # It fell on a Sunday, and the gazette does not count it as a separate holiday.
    assert "Public holiday: no" in answer


def test_2025_deepavali_is_the_state_date() -> None:
    answer = festival_answer("e varshada deepavali yavaga", now=at(2025, 6, 1))

    assert answer is not None
    assert "2025-10-22" in answer


def test_a_kannada_question_in_2025_gets_kannada() -> None:
    answer = festival_answer("2025 ರಲ್ಲಿ ಯುಗಾದಿ ಯಾವಾಗ", now=at(2025, 1, 1))

    assert answer is not None
    assert "ಯುಗಾದಿ" in answer
    assert "2025-03-30" in answer


def test_a_year_with_no_gazette_is_named_rather_than_guessed() -> None:
    answer = festival_answer("2027 ugadi date", now=at(2026, 10, 27))

    assert answer is not None
    assert "2027" in answer
    # No invented festival date: the year is not covered, so no Ugadi date appears.
    assert "2027-04" not in answer
    assert "not published yet" in answer


def test_an_uncovered_year_still_answers_the_dates_that_cannot_move() -> None:
    answer = festival_answer("2027 republic day", now=at(2026, 10, 27))

    assert answer is not None
    assert "2027-01-26" in answer
    assert "2027-08-15" in answer
    assert "2027-10-02" in answer
    assert "2027-11-01" in answer


@pytest.mark.parametrize("year", [2024, 2028, 2030])
def test_boundary_years_do_not_crash(year: int) -> None:
    answer = festival_answer(f"{year} holidays", now=at(2026, 10, 27))

    assert answer is not None
    assert str(year) in answer
    assert f"{year}-01-26" in answer


def test_a_2027_kannada_question_says_the_same_thing() -> None:
    answer = festival_answer("2027 ರಲ್ಲಿ ದೀಪಾವಳಿ ಯಾವಾಗ", now=at(2026, 10, 27))

    assert answer is not None
    assert "2027" in answer
    assert "ಪ್ರಕಟವಾಗಿಲ್ಲ" in answer


def test_a_year_alone_is_not_enough_to_hijack_ordinary_talk() -> None:
    # "2026" appears, but nothing about a festival or a holiday.
    assert festival_answer("2026 was a hard year for me", now=at(2026, 10, 27)) is None
