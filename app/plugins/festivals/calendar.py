"""A festival and holiday answer that costs no tokens and invents nothing.

Festival dates cannot be computed - they follow a lunar calendar and a state gazette - so
they live in per-year data files with their source recorded, and the answer is read from
them. A model asked for "when is Deepavali" will produce a confident date; this will
produce the date in the file, or say that the year is not covered yet.

Years are discovered from the files rather than hard-coded: `data/karnataka_calendar_YYYY.json`
is read at first use, so adding a year is adding a file. That matters because the Karnataka
gazette is published in November or December of the *preceding* year - the 2027 list does not
exist yet, and inventing it would be worse than saying so.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

INDIA = ZoneInfo("Asia/Kolkata")

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_DATA_PATTERN = "karnataka_calendar_*.json"

# Holidays fixed by statute: the same date every year, in every year. These can be answered
# for a year the gazette has not reached yet, because they cannot move.
_FIXED_NATIONAL = (
    {"month": 1, "day": 26, "name_en": "Republic Day", "name_kn": "ಗಣರಾಜ್ಯೋತ್ಸವ"},
    {"month": 8, "day": 15, "name_en": "Independence Day", "name_kn": "ಸ್ವಾತಂತ್ರ್ಯ ದಿನ"},
    {"month": 10, "day": 2, "name_en": "Gandhi Jayanti", "name_kn": "ಗಾಂಧಿ ಜಯಂತಿ"},
    {"month": 11, "day": 1, "name_en": "Kannada Rajyotsava", "name_kn": "ಕನ್ನಡ ರಾಜ್ಯೋತ್ಸವ"},
)

# How people actually ask, in English, Kannada and Kanglish.
_TODAY = re.compile(r"\b(today|ivattu|ivathu|inda)\b|ಇಂದು|ಈ\s*ದಿನ", re.IGNORECASE)
_TOMORROW = re.compile(r"\b(tomorrow|naale|nale)\b|ನಾಳೆ", re.IGNORECASE)
_WEEK = re.compile(r"\b(this\s+week|e\s*vara|ee\s*vara)\b|ಈ\s*ವಾರ", re.IGNORECASE)
_MONTH = re.compile(r"\b(this\s+month|e\s*tingalu|ee\s*tingalu)\b|ಈ\s*ತಿಂಗಳು", re.IGNORECASE)
_EVENT_WORD = re.compile(
    r"\b(festival|festivals|holiday|holidays|habba|habbha|function)\b|"
    r"function\s*ede|ರಜೆ|ಹಬ್ಬ|ಹಬ್ಬಗಳು|ಸಮಾರಂಭ",
    re.IGNORECASE,
)
_WHEN = re.compile(r"\b(when|yavaga)\b|yavaga\s*ide|ಯಾವಾಗ", re.IGNORECASE)
_YEAR = re.compile(r"\b(20\d{2})\b")

# Names as people type them, mapped to the name in the file. Only aliases that mean the
# same festival - never a guess at a different one.
_ALIASES = {
    "deepavali": "deepavali",
    "diwali": "deepavali",
    "ದೀಪಾವಳಿ": "deepavali",
    "ugadi": "ugadi",
    "ಯುಗಾದಿ": "ugadi",
    "ganesh": "ganesh chaturthi",
    "vinayaka": "ganesh chaturthi",
    "ಗಣೇಶ": "ganesh chaturthi",
    "dasara": "vijayadasami",
    "dussehra": "vijayadasami",
    "vijayadasami": "vijayadasami",
    "ವಿಜಯದಶಮಿ": "vijayadasami",
    "rajyotsava": "kannada rajyotsava",
    "ಕನ್ನಡ ರಾಜ್ಯೋತ್ಸವ": "kannada rajyotsava",
    "christmas": "christmas",
    "ಕ್ರಿಸ್ಮಸ್": "christmas",
    "sankranti": "makara sankranti",
    "ಸಂಕ್ರಾಂತಿ": "makara sankranti",
    "bakrid": "bakrid",
    "ಬಕ್ರೀದ್": "bakrid",
    "republic": "republic day",
    "ಗಣರಾಜ್ಯೋತ್ಸವ": "republic day",
    "independence": "independence day",
    "ಸ್ವಾತಂತ್ರ್ಯ": "independence day",
    "gandhi": "gandhi jayanti",
    "ಗಾಂಧಿ": "gandhi jayanti",
    "muharram": "muharram",
    "milad": "eid-milad",
    "id-ul-fitr": "id-ul-fitr",
    "ramzan": "id-ul-fitr",
}


@lru_cache(maxsize=1)
def _by_year() -> dict[int, tuple[dict, ...]]:
    """Every calendar file, indexed by the year it covers. A bad file is skipped, not fatal."""
    calendars: dict[int, list[dict]] = {}

    for path in sorted(_DATA_DIR.glob(_DATA_PATTERN)):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        events = raw.get("events") if isinstance(raw, dict) else None

        if not isinstance(events, list):
            continue

        for item in events:
            if not isinstance(item, dict) or not item.get("date"):
                continue

            try:
                year = int(str(item["date"])[:4])
            except ValueError:
                continue

            calendars.setdefault(year, []).append(item)

    return {year: tuple(items) for year, items in calendars.items()}


def covered_years() -> tuple[int, ...]:
    """The years this calendar can answer for, ascending."""
    return tuple(sorted(_by_year()))


def resolve_year(text: str, today: date) -> int:
    """The year a question is about: an explicit one if it names one, else this one."""
    match = _YEAR.search(text or "")

    if match:
        return int(match.group(1))

    return today.year


def _events(year: int | None = None) -> tuple[dict, ...]:
    calendars = _by_year()

    if year is None:
        return tuple(item for items in calendars.values() for item in items)

    return calendars.get(year, ())


def _parse(item: dict) -> tuple[date, dict] | None:
    try:
        return datetime.strptime(str(item["date"]), "%Y-%m-%d").date(), item
    except (ValueError, KeyError):
        return None


def events_on(day: date) -> list[dict]:
    return [item for item in _events(day.year) if item.get("date") == day.isoformat()]


def events_between(start: date, end: date) -> list[dict]:
    found = []

    for year in range(start.year, end.year + 1):
        for item in _events(year):
            parsed = _parse(item)

            if parsed and start <= parsed[0] <= end:
                found.append(item)

    return sorted(found, key=lambda item: str(item.get("date")))


def find_by_name(query: str, *, year: int | None = None) -> list[dict]:
    """Every event whose name matches, in the year asked about - never a guess."""
    text = query.casefold().strip()

    if not text:
        return []

    wanted = {target for alias, target in _ALIASES.items() if alias in text}
    matches = []

    for item in _events(year):
        name = f"{item.get('name_en', '')} {item.get('name_kn', '')}".casefold()
        key = str(item.get("name_en", "")).casefold()

        if any(target and target in name for target in wanted) or (text and text in name):
            matches.append(item)
        elif key and key in text:
            matches.append(item)

    return sorted(matches, key=lambda item: str(item.get("date")))


def _describe(item: dict, *, kannada: bool) -> str:
    day = str(item.get("date", ""))
    name = item.get("name_kn") if kannada else item.get("name_en")
    holiday = "ಹೌದು" if item.get("is_public_holiday") else "ಇಲ್ಲ"

    if kannada:
        return f"{name}\nದಿನಾಂಕ: {day}\nಸರ್ಕಾರಿ ರಜೆ: {holiday}"

    return f"{name}\nDate: {day}\nPublic holiday: {'yes' if item.get('is_public_holiday') else 'no'}"


def _fixed_dates(year: int, *, kannada: bool) -> str:
    """The holidays that cannot move, so they can be stated for any year."""
    lines = []

    for item in _FIXED_NATIONAL:
        day = date(year, item["month"], item["day"]).isoformat()
        name = item["name_kn"] if kannada else item["name_en"]

        if kannada:
            lines.append(f"{name}\nದಿನಾಂಕ: {day}\nಸರ್ಕಾರಿ ರಜೆ: ಹೌದು")
        else:
            lines.append(f"{name}\nDate: {day}\nPublic holiday: yes")

    return "\n\n".join(lines)


def _uncovered_answer(year: int, *, kannada: bool) -> str:
    """What to say for a year with no gazette in the calendar yet.

    It names the year, gives the holidays that cannot move, and says plainly that the rest
    are not published. A plausible-looking festival date for an unpublished year would be
    indistinguishable from a real one.
    """
    if kannada:
        return (
            f"{year} ರ ಕರ್ನಾಟಕ ರಜಾ ಪಟ್ಟಿ ಇನ್ನೂ ಪ್ರಕಟವಾಗಿಲ್ಲ.\n\n"
            "ಸ್ಥಿರ ದಿನಾಂಕದ ರಜೆಗಳು:\n\n" + _fixed_dates(year, kannada=True) + "\n\n"
            "ಉಳಿದ ಹಬ್ಬಗಳ ದಿನಾಂಕಗಳು ಸರ್ಕಾರದ ಗೆಜೆಟ್ ಪ್ರಕಟವಾದ ನಂತರ ಸೇರಿಸಲಾಗುತ್ತದೆ."
        )

    return (
        f"The Karnataka holiday list for {year} is not published yet, so it is not in this "
        "calendar.\n\nHolidays that cannot move:\n\n"
        + _fixed_dates(year, kannada=False)
        + "\n\nThe festival dates are added once the gazette is out."
    )


def festival_answer(message: str, *, now: datetime | None = None) -> str | None:
    """The answer for a festival or holiday question, or None to let the model handle it.

    None is returned unless the question is clearly about the calendar: a false positive
    here would answer a different question with a date.
    """
    text = (message or "").strip()

    if not text:
        return None

    today = (now or datetime.now(INDIA)).astimezone(INDIA).date()
    kannada = bool(re.search(r"[\u0C80-\u0CFF]", text))
    year = resolve_year(text, today)
    covered = year in _by_year()

    asks_when = bool(_WHEN.search(text))
    asks_event = bool(_EVENT_WORD.search(text))
    named = find_by_name(text, year=year) if covered else []

    if named and (asks_when or asks_event or len(text.split()) <= 6):
        lines = [_describe(item, kannada=kannada) for item in named[:4]]
        return "\n\n".join(lines)

    if not covered:
        # Only take the question when it is about the calendar at all.
        if asks_when or asks_event or _YEAR.search(text):
            return _uncovered_answer(year, kannada=kannada)

        return None

    if _TODAY.search(text) and (asks_event or asks_when):
        return _for_days(today, today, kannada=kannada, empty="ಇಂದು ಯಾವುದೇ ಹಬ್ಬ ಅಥವಾ ರಜೆ ಇಲ್ಲ." if kannada else "No festival or holiday today.")

    if _TOMORROW.search(text) and (asks_event or asks_when):
        day = today + timedelta(days=1)
        return _for_days(day, day, kannada=kannada, empty="ನಾಳೆ ಯಾವುದೇ ಹಬ್ಬ ಅಥವಾ ರಜೆ ಇಲ್ಲ." if kannada else "No festival or holiday tomorrow.")

    if _WEEK.search(text) and (asks_event or asks_when):
        return _for_days(today, today + timedelta(days=6), kannada=kannada, empty="ಈ ವಾರ ಯಾವುದೇ ಹಬ್ಬ ಅಥವಾ ರಜೆ ಇಲ್ಲ." if kannada else "No festival or holiday this week.")

    if _MONTH.search(text) and (asks_event or asks_when):
        end = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        return _for_days(today, end, kannada=kannada, empty="ಈ ತಿಂಗಳು ಯಾವುದೇ ಹಬ್ಬ ಅಥವಾ ರಜೆ ಇಲ್ಲ." if kannada else "No festival or holiday this month.")

    return None


def _for_days(start: date, end: date, *, kannada: bool, empty: str) -> str:
    found = events_between(start, end)

    if not found:
        return empty

    if kannada:
        head = "ಇಂದು:" if start == end == datetime.now(INDIA).date() else f"{start.isoformat()} ರಿಂದ {end.isoformat()} ವರೆಗೆ:"
    else:
        head = "Today:" if start == end == datetime.now(INDIA).date() else f"{start.isoformat()} to {end.isoformat()}:"

    return head + "\n\n" + "\n\n".join(_describe(item, kannada=kannada) for item in found[:8])
