"""Agent 13 - the calendar supervisor: dates are arithmetic, not memory.

A model asked for the fourth Saturday of next month will produce a confident wrong answer,
and a model asked whether Deepavali is a gazetted holiday will sometimes invent a day off.
Neither is acceptable: one makes people miss a bank, the other makes them take leave that
does not exist.

So every date question is answered here, from the state's own notification file, in
Asia/Kolkata:

* **A date is a date.** ``datetime`` and ``zoneinfo`` decide what "tomorrow" means, not the
  model's sense of today.
* **A holiday is a record.** The gazette says which days are general holidays and which are
  occasions that are not. A festival missing from the file is reported as missing from the
  file - never guessed at.
* **A bank holiday is a rule, not a list.** The second and fourth Saturdays are holidays
  under the Negotiable Instruments Act whether or not a festival falls on them, so they are
  computed rather than looked up.

The distinction from agent 05 matters: that agent recognises a holiday *question* and sends
it to the calendar with zero tokens. This one answers the harder questions - how many working
days between two dates, which Saturdays are off, what the month looks like - without a model
being involved at all.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

try:
    from zoneinfo import ZoneInfo

    _KOLKATA = ZoneInfo("Asia/Kolkata")
except Exception:  # a system without tzdata still gets correct local arithmetic
    _KOLKATA = None  # type: ignore[assignment]

MAX_DAYS_AHEAD = 90
MAX_LIMIT = 15
YEAR_RANGE = 2

GAZETTED = "GAZETTED"
BANK = "BANK"
RESTRICTED = "RESTRICTED"

_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# What people type, mapped to an offset in days. The native-script forms are the ones this
# product is actually asked in.
_RELATIVE = {
    "today": 0, "now": 0, "ivattu": 0, "ivathu": 0, "aaj": 0, "ఈరోజు": 0,
    "இன்று": 0, "আজ": 0, "ഇന്ന്": 0, "આજે": 0, "ಇಂದು": 0,
    "tomorrow": 1, "naale": 1, "nale": 1, "kal": 1, "repu": 1, "రేపు": 1,
    "நாளை": 1, "কাল": 1, "നാളെ": 1, "કાલે": 1, "ನಾಳೆ": 1,
    "yesterday": -1, "ninne": -1, "ninnе": -1, "kal tha": -1, "నిన్న": -1,
    "நேற்று": -1, "গতকাল": -1, "ഇന്നലെ": -1, "ગઈકાલે": -1, "ನಿನ್ನೆ": -1,
}

_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_MESSAGES = {
    "INVALID_ARGUMENT": {
        "kn": "ದಿನಾಂಕ ಸರಿಯಾಗಿಲ್ಲ. YYYY-MM-DD ರೂಪದಲ್ಲಿ ಕೊಡಿ.",
        "hi": "तारीख सही नहीं है। YYYY-MM-DD के रूप में दें।",
        "en": "That date is not valid. Please give it as YYYY-MM-DD.",
    },
    "OUT_OF_RANGE": {
        "kn": "ಈ ವರ್ಷದ ದಾಖಲೆ ನನ್ನ ಬಳಿ ಇಲ್ಲ.",
        "hi": "इस वर्ष का रिकॉर्ड मेरे पास नहीं है।",
        "en": "I do not have the notification for that year.",
    },
    "UNKNOWN_TOOL": {
        "kn": "ಆ ವಿನಂತಿ ನನಗೆ ಗೊತ್ತಿಲ್ಲ.",
        "hi": "वह अनुरोध मुझे ज्ञात नहीं है।",
        "en": "I do not know that request.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "calendar_check_date",
        "description": "Whether a date is a holiday, its day of the week, and its category.",
        "parameters": {
            "type": "object",
            "properties": {"target_date": {"type": "string", "description": "YYYY-MM-DD or today/tomorrow"}},
            "required": ["target_date"],
        },
    },
    {
        "name": "calendar_get_upcoming_holidays",
        "description": "The next public and bank holidays, in order.",
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "default": 30},
                "limit": {"type": "integer", "default": 5},
            },
            "required": [],
        },
    },
    {
        "name": "calendar_get_month_schedule",
        "description": "A month: working days, Sundays, second and fourth Saturdays, holidays.",
        "parameters": {
            "type": "object",
            "properties": {"year": {"type": "integer"}, "month": {"type": "integer"}},
            "required": ["year", "month"],
        },
    },
    {
        "name": "calendar_date_difference",
        "description": "Working days between two dates, and how many were holidays.",
        "parameters": {
            "type": "object",
            "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}},
            "required": ["start_date", "end_date"],
        },
    },
)

_REQUIRED = {
    "calendar_check_date": ("target_date",),
    "calendar_get_upcoming_holidays": (),
    "calendar_get_month_schedule": ("year", "month"),
    "calendar_date_difference": ("start_date", "end_date"),
}

_CHIPS_CHECK = (
    {"label": "Next Holiday", "action": "next_holiday"},
    {"label": "Month View", "action": "month_schedule"},
)


@dataclass(slots=True)
class CalendarToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class HolidayRecord:
    date: str
    day_of_week: str
    name_kannada: str
    name_english: str
    category: str


@dataclass(slots=True)
class CalendarSupervisorResult:
    success: bool
    data: Any = None
    formatted_summary: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_CALENDAR_SUPERVISOR_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_tool_declarations() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- time


def _now(now_dt: datetime | None = None) -> datetime:
    if now_dt is not None:
        return now_dt
    return datetime.now(_KOLKATA) if _KOLKATA is not None else datetime.now()


def resolve_relative_date(expression: str, base_dt: datetime | None = None) -> str | None:
    """Turn what a person says into a date, or say nothing rather than guess."""
    text = (expression or "").strip().casefold()

    if not text:
        return None

    if _ISO.match(text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None

    base = _now(base_dt).date()

    if text in _RELATIVE:
        return (base + timedelta(days=_RELATIVE[text])).isoformat()

    if text in {"next week", "nextweek", "ಮುಂದಿನ ವಾರ"}:
        return (base + timedelta(days=7)).isoformat()

    if text in {"next month", "ಮುಂದಿನ ತಿಂಗಳು"}:
        year = base.year + (1 if base.month == 12 else 0)
        month = 1 if base.month == 12 else base.month + 1
        return date(year, month, 1).isoformat()

    weekday = re.match(r"^(?:next\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", text)

    if weekday:
        target = _DAY_NAMES.index(weekday.group(1).capitalize())
        ahead = (target - base.weekday()) % 7 or 7
        return (base + timedelta(days=ahead)).isoformat()

    return None


# ---------------------------------------------------------------- the gazette


def _events(start: date, end: date) -> list[dict[str, Any]]:
    """The state's records in a range, from the one module that owns the file."""
    from app.plugins.festivals import calendar as gazette

    return list(gazette.events_between(start, end))


def _covered_years() -> tuple[int, ...]:
    from app.plugins.festivals import calendar as gazette

    return tuple(gazette.covered_years())


def _category(item: dict[str, Any]) -> str:
    """GAZETTED when the state gave the day off; RESTRICTED when it is an occasion only."""
    return GAZETTED if item.get("is_public_holiday") else RESTRICTED


def _record(item: dict[str, Any]) -> HolidayRecord:
    day = date.fromisoformat(str(item["date"]))
    return HolidayRecord(
        date=day.isoformat(),
        day_of_week=_DAY_NAMES[day.weekday()],
        name_kannada=str(item.get("name_kn", "")),
        name_english=str(item.get("name_en", "")),
        category=_category(item),
    )


def _is_bank_holiday(day: date) -> bool:
    """The second and fourth Saturdays are bank holidays by rule, not by notification."""
    if day.weekday() == 6:  # Sunday
        return True
    if day.weekday() != 5:  # not a Saturday
        return False
    return (day.day - 1) // 7 + 1 in {2, 4}


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date | None:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (nth - 1) * 7
    try:
        return date(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------- validation


def _parse_day(value: Any, base: datetime) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    resolved = resolve_relative_date(str(value or ""), base)
    if resolved is None:
        return None
    return date.fromisoformat(resolved)


def _in_range(day: date, base: datetime) -> bool:
    return abs(day.year - base.year) <= YEAR_RANGE


def _error(code: str, language: str, detail: str = "") -> CalendarSupervisorResult:
    table = _MESSAGES.get(code, _MESSAGES["INVALID_ARGUMENT"])
    key = (language or "en").strip().casefold()[:2]
    message = table.get(key, table["en"])
    return CalendarSupervisorResult(
        success=False,
        formatted_summary=f"{message} {detail}".strip(),
        error_code=code,
        error_message=message,
    )


def validate_arguments(call: CalendarToolCall) -> tuple[dict[str, Any] | None, str | None]:
    if call.tool_name not in _REQUIRED:
        return None, "unknown tool"

    arguments = dict(call.arguments or {})

    for name in _REQUIRED[call.tool_name]:
        if arguments.get(name) is None or str(arguments.get(name)).strip() == "":
            return None, f"missing argument {name!r}"

    if "days_ahead" in arguments:
        try:
            arguments["days_ahead"] = max(1, min(MAX_DAYS_AHEAD, int(arguments["days_ahead"])))
        except (TypeError, ValueError):
            arguments["days_ahead"] = 30

    if "limit" in arguments:
        try:
            arguments["limit"] = max(1, min(MAX_LIMIT, int(arguments["limit"])))
        except (TypeError, ValueError):
            arguments["limit"] = 5

    return arguments, None


# ---------------------------------------------------------------- the tools


def _check_date(day: date, base: datetime, language: str) -> CalendarSupervisorResult:
    if not _in_range(day, base):
        return _error("OUT_OF_RANGE", language, f"({day.year})")

    records = [_record(item) for item in _events(day, day)]
    gazetted = [record for record in records if record.category == GAZETTED]
    bank = _is_bank_holiday(day)

    if gazetted:
        summary = f"{day.isoformat()} ({_DAY_NAMES[day.weekday()]}): {gazetted[0].name_english} — gazetted holiday."
    elif records:
        summary = (
            f"{day.isoformat()} ({_DAY_NAMES[day.weekday()]}): {records[0].name_english} "
            "— an occasion, not a gazetted day off."
        )
    elif bank:
        summary = f"{day.isoformat()} ({_DAY_NAMES[day.weekday()]}): bank holiday."
    else:
        summary = f"{day.isoformat()} ({_DAY_NAMES[day.weekday()]}): a working day."

    return CalendarSupervisorResult(
        success=True,
        data={
            "date": day.isoformat(),
            "day_of_week": _DAY_NAMES[day.weekday()],
            "is_holiday": bool(gazetted),
            "is_bank_holiday": bank,
            "events": [record.__dict__ if hasattr(record, "__dict__") else record for record in []] or [
                {
                    "date": record.date,
                    "day_of_week": record.day_of_week,
                    "name_kannada": record.name_kannada,
                    "name_english": record.name_english,
                    "category": record.category,
                }
                for record in records
            ],
        },
        formatted_summary=summary,
        action_chips=[dict(chip) for chip in _CHIPS_CHECK],
    )


def _upcoming(base: datetime, days_ahead: int, limit: int, language: str) -> CalendarSupervisorResult:
    start = base.date()
    end = start + timedelta(days=days_ahead)

    if not _in_range(end, base):
        return _error("OUT_OF_RANGE", language)

    holidays: list[HolidayRecord] = []

    for item in _events(start, end):
        if item.get("is_public_holiday"):
            holidays.append(_record(item))

    # The Saturday rule fills gaps the gazette does not list, because it does not have to:
    # a bank closes on the second and fourth Saturday whether or not anything is celebrated.
    day = start
    while day <= end:
        if _is_bank_holiday(day) and day.weekday() == 5:
            holidays.append(
                HolidayRecord(
                    date=day.isoformat(),
                    day_of_week=_DAY_NAMES[day.weekday()],
                    name_kannada="ಬ್ಯಾಂಕ್ ರಜೆ",
                    name_english="Bank holiday",
                    category=BANK,
                )
            )
        day += timedelta(days=1)

    holidays.sort(key=lambda record: record.date)
    chosen = holidays[:limit]

    lines = [
        f"{record.date} ({record.day_of_week[:3]}): {record.name_english}"
        for record in chosen
    ]
    summary = "\n".join(lines) if lines else f"No holidays in the next {days_ahead} days."

    return CalendarSupervisorResult(
        success=True,
        data=[
            {
                "date": record.date,
                "day_of_week": record.day_of_week,
                "name_kannada": record.name_kannada,
                "name_english": record.name_english,
                "category": record.category,
            }
            for record in chosen
        ],
        formatted_summary=summary,
        action_chips=[{"label": "Month View", "action": "month_schedule"}],
    )


def _month_schedule(year: int, month: int, base: datetime, language: str) -> CalendarSupervisorResult:
    try:
        first = date(year, month, 1)
    except ValueError:
        return _error("INVALID_ARGUMENT", language)

    if not _in_range(first, base):
        return _error("OUT_OF_RANGE", language, f"({year})")

    last = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1) - timedelta(days=1)

    sundays = sum(1 for day in range(1, last.day + 1) if date(year, month, day).weekday() == 6)
    second = _nth_weekday_of_month(year, month, 5, 2)
    fourth = _nth_weekday_of_month(year, month, 5, 4)

    gazetted = [record for record in (_record(item) for item in _events(first, last)) if record.category == GAZETTED]
    restricted = [record for record in (_record(item) for item in _events(first, last)) if record.category == RESTRICTED]

    holidays: set[date] = {date.fromisoformat(record.date) for record in gazetted}
    for day in range(1, last.day + 1):
        current = date(year, month, day)
        if _is_bank_holiday(current):
            holidays.add(current)

    working = last.day - len(holidays)

    summary = (
        f"{year}-{month:02d}: {last.day} days, {working} working, {sundays} Sundays, "
        f"{len(gazetted)} gazetted holidays"
    )

    return CalendarSupervisorResult(
        success=True,
        data={
            "year": year,
            "month": month,
            "total_days": last.day,
            "working_days": working,
            "sundays": sundays,
            "second_saturday": second.isoformat() if second else None,
            "fourth_saturday": fourth.isoformat() if fourth else None,
            "gazetted": [record.date for record in gazetted],
            "restricted": [record.date for record in restricted],
        },
        formatted_summary=summary,
        action_chips=[{"label": "Next Holiday", "action": "next_holiday"}],
    )


def _difference(start: date, end: date, base: datetime, language: str) -> CalendarSupervisorResult:
    if end < start:
        return _error("INVALID_ARGUMENT", language)

    if not (_in_range(start, base) and _in_range(end, base)):
        return _error("OUT_OF_RANGE", language)

    gazetted = {record.date for record in (_record(item) for item in _events(start, end)) if record.category == GAZETTED}

    total = (end - start).days + 1
    working = 0
    holiday_days = 0

    for offset in range(total):
        day = start + timedelta(days=offset)
        if day.isoformat() in gazetted or _is_bank_holiday(day):
            holiday_days += 1
        else:
            working += 1

    return CalendarSupervisorResult(
        success=True,
        data={
            "start": start.isoformat(),
            "end": end.isoformat(),
            "total_days": total,
            "working_days": working,
            "holiday_days": holiday_days,
        },
        formatted_summary=(
            f"{start.isoformat()} to {end.isoformat()}: {total} days, "
            f"{working} working, {holiday_days} holidays"
        ),
        action_chips=[],
    )


# ---------------------------------------------------------------- the decision


def validate_and_execute(
    call: CalendarToolCall,
    now_dt: datetime | None = None,
    *,
    language: str = "en",
) -> CalendarSupervisorResult:
    """Answer a date question from the gazette and the calendar, never from memory."""
    started = time.perf_counter()
    base = _now(now_dt)

    arguments, problem = validate_arguments(call)

    if problem is not None:
        code = "UNKNOWN_TOOL" if problem == "unknown tool" else "INVALID_ARGUMENT"
        result = _error(code, language)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    assert arguments is not None

    if call.tool_name == "calendar_check_date":
        day = _parse_day(arguments["target_date"], base)
        if day is None:
            result = _error("INVALID_ARGUMENT", language)
        else:
            result = _check_date(day, base, language)

    elif call.tool_name == "calendar_get_upcoming_holidays":
        result = _upcoming(
            base,
            int(arguments.get("days_ahead", 30)),
            int(arguments.get("limit", 5)),
            language,
        )

    elif call.tool_name == "calendar_get_month_schedule":
        try:
            year = int(arguments["year"])
            month = int(arguments["month"])
        except (TypeError, ValueError):
            result = _error("INVALID_ARGUMENT", language)
        else:
            result = _month_schedule(year, month, base, language)

    else:
        start = _parse_day(arguments["start_date"], base)
        end = _parse_day(arguments["end_date"], base)

        if start is None or end is None:
            result = _error("INVALID_ARGUMENT", language)
        else:
            result = _difference(start, end, base, language)

    result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
    return result
