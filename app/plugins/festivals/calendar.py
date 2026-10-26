"""A festival and holiday answer that costs no tokens and invents nothing.

Festival dates cannot be computed - they follow a lunar calendar and a state gazette - so
they live in a data file with its source recorded, and the answer is read from it. A model
asked for "when is Deepavali" will produce a confident date; this will produce the date in
the file, or say that the calendar does not cover the year yet.

The router runs before any model gateway, so a festival question costs nothing.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

INDIA = ZoneInfo("Asia/Kolkata")

_DATA_FILE = Path(__file__).resolve().parents[3] / "data" / "karnataka_calendar_2026.json"

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
def _events() -> tuple[dict, ...]:
    """The calendar, read once. A missing file is an empty calendar, not a crash."""
    try:
        raw = json.loads(_DATA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()

    events = raw.get("events") if isinstance(raw, dict) else None

    if not isinstance(events, list):
        return ()

    return tuple(item for item in events if isinstance(item, dict) and item.get("date"))


def _parse(item: dict) -> tuple[date, dict] | None:
    try:
        return datetime.strptime(str(item["date"]), "%Y-%m-%d").date(), item
    except (ValueError, KeyError):
        return None


def events_on(day: date) -> list[dict]:
    return [item for item in _events() if item.get("date") == day.isoformat()]


def events_between(start: date, end: date) -> list[dict]:
    found = []

    for item in _events():
        parsed = _parse(item)

        if parsed and start <= parsed[0] <= end:
            found.append(item)

    return sorted(found, key=lambda item: str(item.get("date")))


def find_by_name(query: str) -> list[dict]:
    """Every event whose name matches, newest name match only - never a guess."""
    text = query.casefold().strip()

    if not text:
        return []

    wanted = {target for alias, target in _ALIASES.items() if alias in text}
    matches = []

    for item in _events():
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
    named = find_by_name(text)
    asks_when = bool(_WHEN.search(text))
    asks_event = bool(_EVENT_WORD.search(text))

    if named and (asks_when or asks_event or len(text.split()) <= 6):
        lines = [_describe(item, kannada=kannada) for item in named[:4]]
        return "\n\n".join(lines)

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
