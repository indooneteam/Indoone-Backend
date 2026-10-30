"""Agent 03 - the Vault Guardian: thinks before it redacts.

A blind replacer that turns every twelve-digit number into "[Aadhaar Redacted]" destroys
commit hashes, timestamps and port numbers, and a user who cannot paste a stack trace will
stop pasting anything. So this agent reads context before it acts:

* a **Unix timestamp** is ten digits beginning with 1 or 2, and is left alone;
* a **git hash** is hexadecimal, so it is neither a PAN (five letters, four digits, one
  letter) nor an Aadhaar (twelve digits), and is left alone;
* a **port or a small number after `=` or `:`** is configuration, not a phone number;
* a **card** is redacted only when its digits pass the Luhn checksum, so an order number
  that happens to be sixteen digits survives.

What it does redact, in the order it looks: infrastructure secrets first (API keys, database
URIs with credentials, JWTs), then Indian identifiers (Aadhaar, PAN, IFSC, UPI), then cards,
phones and email addresses, then the global identifiers (SSN, Korean RRN, Japanese MyNumber
when it is named as one).

The contract is typed: PASS or REDACTED, the original, the redacted text, which entities were
found, how many, and telemetry. Standard library only, pre-compiled patterns, bounded work.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

PASS = "PASS"
REDACTED = "REDACTED"

# Placeholders. Spelled out rather than symbols so a reader of the transcript knows what
# was removed without seeing what it was.
PLACEHOLDERS = {
    "api_key": "[API Key Redacted]",
    "aws_key": "[AWS Key Redacted]",
    "github_token": "[GitHub Token Redacted]",
    "jwt": "[Token Redacted]",
    "db_credentials": "[Credentials Redacted]",
    "aadhaar": "[Aadhaar Redacted]",
    "pan": "[PAN Redacted]",
    "ifsc": "[IFSC Redacted]",
    "upi": "[UPI Redacted]",
    "card": "[Card Redacted]",
    "phone": "[Phone Redacted]",
    "email": "[Email Redacted]",
    "ssn": "[SSN Redacted]",
    "rrn": "[RRN Redacted]",
    "mynumber": "[MyNumber Redacted]",
}

# ---------------------------------------------------------------- patterns

_API_KEYS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "api_key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"), "github_token"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "github_token"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws_key"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"), "api_key"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "api_key"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"), "jwt"),
)

# scheme://user:password@host - only the credentials go, so the shape stays readable.
_DB_URI = re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://[^\s:/@]+:([^\s@]+)@", re.IGNORECASE)

_AADHAAR = re.compile(r"(?<![\d])([2-9]\d{3})[\s-]?(\d{4})[\s-]?(\d{4})(?![\d])")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_IFSC = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")
_UPI = re.compile(r"\b[\w.+-]{2,}@(okaxis|oksbi|okhdfcbank|okicici|okbizaxis|upi|paytm|ybl|ibl|axl|apl)\b", re.IGNORECASE)
_CARD = re.compile(r"(?<![\d])(?:\d[ -]?){13,19}(?![\d])")
# People write a mobile number in groups: "+91 98765 43210", "098765-43210", "9876543210".
# Three shapes, because people group a mobile number three ways: 9876543210,
# 98765 43210 and 987 654 3210 - each optionally behind +91 or a trunk 0.
_PHONE = re.compile(
    r"(?<![\d])(?:\+?91[\s-]?|0)?"
    r"([6-9]\d{9}|[6-9]\d{4}[\s-]\d{5}|[6-9]\d{2}[\s-]\d{3}[\s-]\d{4})"
    r"(?![\d])"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"(?<![\d])\d{3}-\d{2}-\d{4}(?![\d])")
_RRN = re.compile(r"(?<![\d])\d{6}-[1-8]\d{6}(?![\d])")
_MYNUMBER = re.compile(r"(?:my\s*number|mynumber|マイナンバー)[^\d]{0,10}(\d{12})", re.IGNORECASE)

# Ten digits starting with 1 or 2 is a Unix timestamp, not a phone number.
_TIMESTAMP = re.compile(r"^[12]\d{9}$")
# Seven to forty hexadecimal characters is a commit or blob hash.
_GIT_HASH = re.compile(r"^[0-9a-f]{7,40}$")
# A short number glued to a port or an assignment is configuration.
_CONFIG_CONTEXT = re.compile(r"(?:port\s*[=:]\s*|[:=]\s*)$", re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class Telemetry:
    execution_microseconds: int
    entities_evaluated: int


@dataclass(slots=True, frozen=True)
class VaultVerdict:
    """PASS when nothing sensitive was found, REDACTED when something was."""

    status: str
    original_prompt: str
    redacted_prompt: str
    detected_entities: list[str]
    redaction_count: int
    telemetry: Telemetry

    @property
    def changed(self) -> bool:
        return self.status == REDACTED


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_VAULT_GUARDIAN_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}


# ---------------------------------------------------------------- reasoning helpers


def _luhn(digits: str) -> bool:
    """The checksum that separates a card number from a sixteen-digit order id."""
    cleaned = re.sub(r"\D", "", digits)

    if not 13 <= len(cleaned) <= 19:
        return False

    total = 0
    for position, character in enumerate(reversed(cleaned)):
        value = int(character)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value

    return total % 10 == 0


def _is_technical_number(text: str, match: re.Match) -> bool:
    """True when a number is code, a hash, a timestamp or configuration - not a person."""
    token = match.group(0).strip()
    digits = re.sub(r"\D", "", token)
    before = text[max(0, match.start() - 12) : match.start()]

    # A timestamp is ten digits and nothing else. Checking the digits of a token that also
    # carries letters is how "sk-abcdefghijklmnop1234567890" was read as a timestamp and
    # left in the prompt.
    if len(digits) == 10 and digits == token and _TIMESTAMP.match(digits):
        return True

    # A hash carries at least one letter. A number is not a hash, however hexadecimal it is.
    if any(character in "abcdef" for character in token.lower()) and _GIT_HASH.match(token.lower()):
        return True

    # port=8000, :5432, timeout=30 - a number that belongs to a setting
    if len(digits) <= 5 and _CONFIG_CONTEXT.search(before):
        return True

    return False


def _replace(
    text: str,
    pattern: re.Pattern,
    placeholder: str,
    entity: str,
    found: list[str],
    *,
    guard: bool = False,
) -> str:
    """Replace every match, unless the numeric guard says it is code.

    ``guard`` is off for the identifier families (PAN, IFSC, UPI): they cannot be confused
    with a timestamp or a hash, and a PAN such as ABCDE1234F is hexadecimal by accident.
    """
    def repl(match: re.Match) -> str:
        if guard and _is_technical_number(text, match):
            return match.group(0)
        found.append(entity)
        return placeholder

    return pattern.sub(repl, text)


def redact(message: str) -> VaultVerdict:
    """Return the same message with what must not leave the machine removed."""
    started = time.perf_counter()
    original = message or ""
    working = original
    found: list[str] = []
    evaluated = 0

    # 1. Infrastructure secrets first: they are the ones that cost money and access.
    for pattern, entity in _API_KEYS:
        evaluated += 1
        working = _replace(working, pattern, PLACEHOLDERS[entity], entity, found)

    evaluated += 1
    def credentials(match: re.Match) -> str:
        found.append("db_credentials")
        return match.group(0).replace(match.group(1), "***", 1)

    working = _DB_URI.sub(credentials, working)

    # 2. Cards first, and only when the checksum agrees: a sixteen-digit card carries a
    # twelve-digit run that would otherwise be taken for an Aadhaar number. A card-shaped
    # number that FAILS the checksum is remembered as a protected span, so the Aadhaar pass
    # does not then take a slice of it - that would redact an order id as an identity number.
    evaluated += 1
    protected: list[tuple[int, int]] = []

    def card(match: re.Match) -> str:
        if not _luhn(match.group(0)):
            protected.append((match.start(), match.end()))
            return match.group(0)
        found.append("card")
        return PLACEHOLDERS["card"]

    working = _CARD.sub(card, working)

    # 3. Indian identifiers.
    evaluated += 1
    def aadhaar(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if _is_technical_number(working, match) or len(set(digits)) == 1:
            return match.group(0)
        if any(start <= match.start() and match.end() <= end for start, end in protected):
            return match.group(0)
        found.append("aadhaar")
        return PLACEHOLDERS["aadhaar"]

    working = _AADHAAR.sub(aadhaar, working)

    for pattern, entity in ((_PAN, "pan"), (_IFSC, "ifsc"), (_UPI, "upi")):
        evaluated += 1
        working = _replace(working, pattern, PLACEHOLDERS[entity], entity, found)


    # 4. Contact data.
    evaluated += 1
    working = _replace(working, _PHONE, PLACEHOLDERS["phone"], "phone", found, guard=True)
    evaluated += 1
    working = _replace(working, _EMAIL, PLACEHOLDERS["email"], "email", found)

    # 5. The global identifiers.
    for pattern, entity in ((_SSN, "ssn"), (_RRN, "rrn")):
        evaluated += 1
        working = _replace(working, pattern, PLACEHOLDERS[entity], entity, found)

    evaluated += 1
    def mynumber(match: re.Match) -> str:
        found.append("mynumber")
        return match.group(0).replace(match.group(1), PLACEHOLDERS["mynumber"], 1)

    working = _MYNUMBER.sub(mynumber, working)

    return VaultVerdict(
        status=REDACTED if found else PASS,
        original_prompt=original,
        redacted_prompt=working,
        detected_entities=sorted(set(found)),
        redaction_count=len(found),
        telemetry=Telemetry(
            execution_microseconds=int((time.perf_counter() - started) * 1_000_000),
            entities_evaluated=evaluated,
        ),
    )
