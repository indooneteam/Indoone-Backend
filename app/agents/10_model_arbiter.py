"""Agent 10 - the Arbiter: the last thing between the model and the person.

Everything before this decides what the model is asked. This decides what the user is given,
and it is the only agent whose mistakes the user sees directly. Three jobs:

**1. Do not leak the scaffolding.** A model sometimes begins by repeating its instructions -
"You are an AI assistant...", "System instructions:", a stray `<prompt_context>` tag. That
text is not an answer, and shipping it tells the user how the machine is built. The shield
removes a leaked preamble and keeps everything after it; if the whole reply is scaffolding,
nothing is shipped rather than a fragment.

**2. Chips the client can act on.** Tools that already emitted chips stay authoritative -
this never invents a suggestion over a real one. When nothing was emitted, a suggestion is
inferred from what the answer is about: a branch reply offers a diff, a holiday reply offers
the next one. A chip is a label and an action, which is the contract `indoone-main` renders.

**3. One answer, four channels.** WhatsApp wants `*bold*` and no tables; Telegram wants
MarkdownV2 and inline buttons; Instagram has 1000 characters and no formatting at all; the
app takes full Markdown. Sending the app's Markdown to Instagram is how a channel shows raw
asterisks to a customer.

Bounded, standard library only, module-level patterns so nothing compiles per call.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

APP = "app"
WHATSAPP = "whatsapp"
TELEGRAM = "telegram"
INSTAGRAM = "instagram"

WHATSAPP_LIMIT = 4096
INSTAGRAM_LIMIT = 1000

# A leaked preamble: the model repeating what it was told. Anchored at the start, because a
# mention of "system instructions" in the middle of a technical answer is content.
_LEAK_PREAMBLE = re.compile(
    r"^\s*(?:"
    r"you\s+are\s+(?:an?\s+)?(?:ai|a\.i\.|helpful|large\s+language)[^\n]*(?:\n|$)"
    r"|system\s+(?:prompt|instruction|directive)s?\s*:?[^\n]*(?:\n|$)"
    r"|<(?:prompt_context|system|instructions?)>[^\n]*\n?"
    r"|as\s+an\s+ai\s+language\s+model[^\n]*(?:\n|$)"
    r")+",
    re.IGNORECASE,
)

_LEAK_ANYWHERE = re.compile(
    r"</?(?:prompt_context|system_prompt|instructions?)>",
    re.IGNORECASE,
)

_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_CODE_FENCE = re.compile(r"```[a-zA-Z0-9]*\n?")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

# What a reply is about, and what a person would want to do next.
_CHIP_RULES: tuple[tuple[re.Pattern, tuple[dict[str, str], ...]], ...] = (
    (
        re.compile(r"\b(branch|branches|ಶಾಖೆ)\b", re.IGNORECASE),
        ({"label": "View Diff", "action": "diff"}, {"label": "Check PRs", "action": "prs"}),
    ),
    (
        re.compile(r"\b(pull request|pr|merge)\b", re.IGNORECASE),
        ({"label": "View Diff", "action": "diff"},),
    ),
    (
        re.compile(r"\b(commit|commits)\b", re.IGNORECASE),
        ({"label": "Recent Commits", "action": "commits"},),
    ),
    (
        re.compile(
            r"\b(holiday|festival|rajyotsava|deepavali|diwali|ugadi|dasara|dussehra|"
            r"vijayadasami|sankranti|christmas|ramzan|eid|ಹಬ್ಬ|ರಜೆ|ದೀಪಾವಳಿ|ಯುಗಾದಿ)\b",
            re.IGNORECASE,
        ),
        ({"label": "Next Holiday", "action": "next_holiday"},),
    ),
    (
        re.compile(r"\b(email|inbox|mail)\b", re.IGNORECASE),
        ({"label": "Unread Mail", "action": "unread_email"},),
    ),
)

_CHIPS_BY_INTENT: dict[str, tuple[dict[str, str], ...]] = {
    "LIST": ({"label": "View Diff", "action": "diff"},),
    "DIFF": ({"label": "Check PRs", "action": "prs"},),
    "CHECK_HOLIDAY": ({"label": "Next Holiday", "action": "next_holiday"},),
    "READ_EMAIL": ({"label": "Unread Mail", "action": "unread_email"},),
}


@dataclass(slots=True)
class SynthesizedTurnResult:
    """The reply, ready for one channel, with what it cost to prepare it."""

    formatted_text: str
    action_chips: list[dict[str, str]] = field(default_factory=list)
    channel_payload: dict[str, object] = field(default_factory=dict)
    leaks_redacted: bool = False
    telemetry_micros: float = 0.0


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_MODEL_ARBITER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


# ---------------------------------------------------------------- 1. the shield


def strip_leaks(text: str) -> tuple[str, bool]:
    """Remove a leaked instruction preamble. Returns the text and whether anything went."""
    body = text or ""

    if not body.strip():
        return body, False

    cleaned = _LEAK_PREAMBLE.sub("", body, count=1)
    cleaned, tags = _LEAK_ANYWHERE.subn("", cleaned)

    redacted = cleaned != body

    # If the whole reply was scaffolding there is nothing to say. An empty answer is honest;
    # a fragment of an instruction is not.
    if redacted and not cleaned.strip():
        return "", True

    return cleaned.lstrip("\n"), redacted or bool(tags)


# ---------------------------------------------------------------- 2. the chips


def synthesize_chips(text: str, intent: str = "", existing: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Follow-up suggestions for this answer.

    A tool that already emitted chips wins: those came from the tool that did the work, and
    a guess must never be offered over a fact.
    """
    if existing:
        return [dict(chip) for chip in existing]

    kind = (intent or "").strip().upper()

    if kind in _CHIPS_BY_INTENT:
        return [dict(chip) for chip in _CHIPS_BY_INTENT[kind]]

    for pattern, chips in _CHIP_RULES:
        if pattern.search(text or ""):
            return [dict(chip) for chip in chips]

    return []


# ---------------------------------------------------------------- 3. the channels


def _tables_to_lists(text: str) -> str:
    """A Markdown table is unreadable where tables do not render."""
    lines = (text or "").splitlines()
    output: list[str] = []
    header: list[str] = []
    in_table = False

    for line in lines:
        if _TABLE_DIVIDER.match(line):
            in_table = True
            continue

        if _TABLE_ROW.match(line):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]

            if not in_table:
                header = cells
            else:
                label = cells[0] if cells else ""
                rest = [cell for cell in cells[1:] if cell]

                if header and len(header) == len(cells):
                    pairs = [
                        f"{header[index]}: {cell}"
                        for index, cell in enumerate(cells[1:], start=1)
                        if cell
                    ]
                    output.append(f"- {label} — " + ", ".join(pairs) if pairs else f"- {label}")
                else:
                    output.append("- " + " — ".join(rest or cells))
            continue

        if in_table and not line.strip():
            in_table = False
            header = []

        output.append(line)

    return "\n".join(output)


# A virama joins the consonant after it, so a slice between them breaks the glyph. The
# danda-like marks and the vowel signs are combining characters, and unicodedata says so.
_VIRAMA = frozenset({"\u0ccd", "\u094d", "\u0c4d", "\u0bcd", "\u09cd", "\u0acd", "\u0d4d"})


def _is_safe_boundary(text: str, index: int) -> bool:
    """Whether cutting before ``index`` leaves both sides renderable.

    Cutting mid-cluster is what produces the boxes and broken conjuncts people see on
    Instagram: a dependent vowel sign with no consonant, or a virama with nothing to join.
    """
    if index <= 0 or index >= len(text):
        return True

    following = text[index]

    if unicodedata.combining(following):
        return False

    if text[index - 1] in _VIRAMA:
        return False

    return True


def _safe_cut(text: str, limit: int) -> int:
    """The largest index at or below ``limit`` that does not split a cluster."""
    cut = limit

    while cut > 0 and not _is_safe_boundary(text, cut):
        cut -= 1

    if cut <= 0:
        return 0

    # Then prefer a word boundary, so the cut reads as an ending rather than a chop.
    for separator in (" ", "\n", "\u0964", "\u0965", "|", "."):
        position = text.rfind(separator, 0, cut)
        if position > cut // 2:
            return position

    return cut


def _split(text: str, limit: int) -> list[str]:
    """Break at a paragraph, then a line, then a word - never inside a word."""
    body = text or ""

    if len(body) <= limit:
        return [body] if body else []

    chunks: list[str] = []
    remaining = body

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")

        if cut < limit // 2:
            cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit

        cut = _safe_cut(remaining, cut) or cut
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _plain(text: str) -> str:
    """Strip the markup, keep the words."""
    body = _CODE_FENCE.sub("", text or "")
    body = _INLINE_CODE.sub(r"\1", body)
    body = _HEADING.sub("", body)
    body = _BOLD.sub(r"\1", body)
    body = _ITALIC.sub(r"\1", body)
    body = _LINK.sub(r"\1 (\2)", body)
    return body


def to_whatsapp(text: str) -> dict[str, object]:
    """Bold with single asterisks, no tables, and messages that fit."""
    body = _tables_to_lists(text)
    body = _BOLD.sub(r"*\1*", body)
    body = _HEADING.sub("*", body)

    chunks = _split(body, WHATSAPP_LIMIT)

    return {"text": chunks[0] if chunks else "", "chunks": chunks, "format": "whatsapp"}


def to_telegram(text: str, chips: list[dict[str, str]] | None = None) -> dict[str, object]:
    """MarkdownV2-safe text and an inline keyboard built from the chips."""
    body = _plain(text)
    body = _tables_to_lists(body)

    keyboard = [
        [{"text": chip["label"], "callback_data": chip["action"]}]
        for chip in (chips or [])
    ]

    return {
        "text": _split(body, WHATSAPP_LIMIT)[0],
        "parse_mode": "MarkdownV2",
        "reply_markup": {"inline_keyboard": keyboard} if keyboard else {},
        "format": "telegram",
    }


def to_instagram(text: str) -> dict[str, object]:
    """One thousand characters, no markup, no tables."""
    body = _plain(_tables_to_lists(text))
    body = " ".join(body.split())

    if len(body) > INSTAGRAM_LIMIT:
        cut = _safe_cut(body, INSTAGRAM_LIMIT - 1)
        body = body[:cut].rstrip() + "…"

    return {"text": body, "format": "instagram", "truncated": len(body) >= INSTAGRAM_LIMIT - 1}


def to_app(text: str, chips: list[dict[str, str]] | None = None) -> dict[str, object]:
    """Full Markdown and the chips as the client expects them."""
    return {
        "text": text or "",
        "action_chips": list(chips or []),
        "format": "markdown",
    }


# ---------------------------------------------------------------- the decision


def synthesize(
    raw_text: str,
    channel: str = APP,
    intent: str = "",
    language: str = "",
    existing_chips: list[dict[str, str]] | None = None,
) -> SynthesizedTurnResult:
    """Clean the answer, decide its chips, and format it for one channel."""
    import time

    started = time.perf_counter()

    cleaned, redacted = strip_leaks(raw_text)
    chips = synthesize_chips(cleaned, intent, existing_chips)

    name = (channel or APP).strip().casefold()

    if name == WHATSAPP:
        payload = to_whatsapp(cleaned)
    elif name == TELEGRAM:
        payload = to_telegram(cleaned, chips)
    elif name == INSTAGRAM:
        payload = to_instagram(cleaned)
    else:
        payload = to_app(cleaned, chips)

    return SynthesizedTurnResult(
        formatted_text=str(payload.get("text") or ""),
        action_chips=chips,
        channel_payload=payload,
        leaks_redacted=redacted,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
