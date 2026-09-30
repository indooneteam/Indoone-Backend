from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ALLOWED_LABELS = {"legitimate", "spam", "phishing"}
MAX_EMAIL_WORDS = 240
MAX_RESPONSE_WORDS = 140
FORBIDDEN_SECRET_PATTERNS = (
    re.compile(
        r"\b(?:otp|one[- ]time|verification|security|pin|passcode|code)\s*"
        r"[:=-]?\s*\d{4,8}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:card|credit|debit)\s*(?:number|no\.?)?\s*[:=-]?\s*"
        r"(?:\d[ -]?){12,19}\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:sk|pk)_[A-Za-z0-9_-]{16,}\b", re.IGNORECASE),
)
LIVE_URL_HINT = re.compile(r"https?://(?![^\s/]*example\.com\b)[^\s]+", re.IGNORECASE)
LABEL_IN_SUBJECT_HINT = re.compile(
    r"\b(?:legitimate|spam|phishing)\b", re.IGNORECASE
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def validate(path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    counts = {label: 0 for label in ALLOWED_LABELS}

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
            continue

        if not isinstance(payload, dict):
            errors.append(f"line {line_number}: record must be an object")
            continue

        instruction = payload.get("instruction")
        response = payload.get("response")
        category = payload.get("category")

        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"line {line_number}: instruction is required")
            continue
        if not isinstance(response, str) or not response.strip():
            errors.append(f"line {line_number}: response is required")
            continue
        if category != "email_safety":
            errors.append(f"line {line_number}: category must be email_safety")

        label_match = re.search(
            r"(?:classification|vargikarana|वर्गीकरण)\s*[:=-]\s*"
            r"(legitimate|spam|phishing)\b",
            response,
            re.IGNORECASE,
        )
        if not label_match:
            errors.append(f"line {line_number}: response must state a primary classification")
            continue

        label = label_match.group(1).casefold()
        counts[label] += 1

        key = (instruction.casefold(), response.casefold())
        if key in seen:
            errors.append(f"line {line_number}: duplicate instruction/response")
        seen.add(key)

        if "EMAIL:" not in instruction.upper():
            errors.append(f"line {line_number}: instruction must include EMAIL:")
        email_part = instruction.split("EMAIL:", 1)[-1]

        if _word_count(email_part) > MAX_EMAIL_WORDS:
            errors.append(f"line {line_number}: email exceeds {MAX_EMAIL_WORDS} words")
        if _word_count(response) > MAX_RESPONSE_WORDS:
            errors.append(f"line {line_number}: response exceeds {MAX_RESPONSE_WORDS} words")

        if LABEL_IN_SUBJECT_HINT.search(email_part.splitlines()[1] if len(email_part.splitlines()) > 1 else ""):
            errors.append(f"line {line_number}: label leakage in email subject")

        for pattern in FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(instruction) or pattern.search(response):
                errors.append(f"line {line_number}: possible secret-like value found")

        if LIVE_URL_HINT.search(instruction):
            errors.append(f"line {line_number}: non-example.com URL found")

    return sum(counts.values()), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Indoone Email Safety JSONL.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    count, errors = validate(args.path)
    if errors:
        for error in errors:
            print(error)
        print(f"validation failed: {len(errors)} error(s)")
        return 1

    print(f"validation passed: {count} email-safety examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
