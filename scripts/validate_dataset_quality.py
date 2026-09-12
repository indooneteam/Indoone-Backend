from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DEFAULT_SOURCES = (
    Path("data/raw/indoone_instructions.jsonl"),
    Path("data/raw/core_instruction_seed.jsonl"),
    Path("data/raw/indoone_multilingual_examples.jsonl"),
)

REQUIRED_FIELDS = ("instruction", "response", "category")
DEFAULT_MIN_RESPONSE_CHARS = 8
DEFAULT_MAX_SHORT_RESPONSE_RATE = 0.05
DEFAULT_MIN_CATEGORY_COUNT = 1


def _normalize(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _read_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _looks_like_placeholder(text: str) -> bool:
    normalized = text.casefold()
    placeholders = (
        "todo",
        "tbd",
        "placeholder",
        "lorem ipsum",
        "insert answer",
        "example response",
    )
    return any(token in normalized for token in placeholders)


def validate_rows(
    rows: list[dict[str, object]],
    *,
    min_response_chars: int = DEFAULT_MIN_RESPONSE_CHARS,
    max_short_response_rate: float = DEFAULT_MAX_SHORT_RESPONSE_RATE,
    min_category_count: int = DEFAULT_MIN_CATEGORY_COUNT,
) -> dict[str, object]:
    failures: list[str] = []
    duplicate_keys: Counter[tuple[str, str]] = Counter()
    categories: Counter[str] = Counter()
    short_count = 0
    placeholder_count = 0
    invalid_count = 0

    for index, row in enumerate(rows, 1):
        missing = [field for field in REQUIRED_FIELDS if not _normalize(row.get(field))]
        if missing:
            invalid_count += 1
            failures.append(f"row {index} missing required fields: {', '.join(missing)}")
            continue
        instruction = _normalize(row["instruction"])
        response = _normalize(row["response"])
        category = _normalize(row["category"]).casefold()
        duplicate_keys[(instruction.casefold(), response.casefold())] += 1
        categories[category] += 1
        if len(response) < min_response_chars:
            short_count += 1
        if _looks_like_placeholder(response):
            placeholder_count += 1

    duplicates = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    duplicate_rate = duplicates / len(rows) if rows else 1.0
    short_rate = short_count / len(rows) if rows else 1.0

    if not rows:
        failures.append("dataset is empty")
    if duplicate_rate > 0:
        failures.append(f"duplicate examples found: {duplicates} ({duplicate_rate:.2%})")
    if short_rate > max_short_response_rate:
        failures.append(
            f"short responses {short_rate:.2%} exceed allowed {max_short_response_rate:.2%}"
        )
    if placeholder_count:
        failures.append(f"placeholder responses found: {placeholder_count}")
    if invalid_count:
        failures.append(f"invalid rows found: {invalid_count}")
    for category, count in categories.items():
        if count < min_category_count:
            failures.append(f"category {category!r} has only {count} example(s)")

    return {
        "ready": not failures,
        "rows": len(rows),
        "categories": dict(categories),
        "duplicates": duplicates,
        "duplicate_rate": duplicate_rate,
        "short_responses": short_count,
        "short_response_rate": short_rate,
        "placeholder_responses": placeholder_count,
        "invalid_rows": invalid_count,
        "failures": failures,
        "rules": {
            "min_response_chars": min_response_chars,
            "max_short_response_rate": max_short_response_rate,
            "min_category_count": min_category_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Indoone training-data quality before training.")
    parser.add_argument("--source", action="append", type=Path, dest="sources")
    parser.add_argument("--min-response-chars", type=int, default=DEFAULT_MIN_RESPONSE_CHARS)
    parser.add_argument("--max-short-response-rate", type=float, default=DEFAULT_MAX_SHORT_RESPONSE_RATE)
    args = parser.parse_args()

    if args.min_response_chars <= 0:
        raise SystemExit("--min-response-chars must be positive")
    if not 0 <= args.max_short_response_rate <= 1:
        raise SystemExit("--max-short-response-rate must be between 0 and 1")

    sources = args.sources or list(DEFAULT_SOURCES)
    report = validate_rows(
        _read_rows(sources),
        min_response_chars=args.min_response_chars,
        max_short_response_rate=args.max_short_response_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
