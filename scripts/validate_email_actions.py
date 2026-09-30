from __future__ import annotations

import argparse
import json
from pathlib import Path

ALLOWED_CATEGORIES = {"email_compose", "email_reply", "email_send"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Indoone email-action training examples.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    seen: set[tuple[str, str]] = set()
    count = 0
    errors: list[str] = []

    for line_number, line in enumerate(args.path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"line {line_number}: invalid JSON")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: record must be an object")
            continue

        instruction = row.get("instruction")
        response = row.get("response")
        category = row.get("category")
        if not isinstance(instruction, str) or not instruction.strip():
            errors.append(f"line {line_number}: instruction required")
        if not isinstance(response, str) or not response.strip():
            errors.append(f"line {line_number}: response required")
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"line {line_number}: invalid category")
        key = (str(instruction).casefold(), str(response).casefold())
        if key in seen:
            errors.append(f"line {line_number}: duplicate example")
        seen.add(key)
        count += 1

    if errors:
        print("
".join(errors))
        print(f"validation failed: {len(errors)} error(s)")
        return 1

    print(f"validation passed: {count} email-action examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
