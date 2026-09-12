from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCES = (
    Path("data/raw/indoone_instructions.jsonl"),
    Path("data/raw/core_instruction_seed.jsonl"),
    Path("data/raw/indoone_multilingual_examples.jsonl"),
)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _fingerprint(row: dict[str, object]) -> str:
    payload = {
        "instruction": _normalize(row.get("instruction")),
        "response": _normalize(row.get("response")),
        "category": _normalize(row.get("category")),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        instruction = _normalize(value.get("instruction"))
        response = _normalize(value.get("response"))
        if not instruction or not response:
            raise ValueError(f"empty instruction/response at {path}:{line_number}")
        yield {
            **value,
            "instruction": instruction,
            "response": response,
            "category": _normalize(value.get("category")) or "uncategorized",
        }


def assemble(
    sources: list[Path],
    output: Path,
    validation_output: Path,
    validation_ratio: float,
    seed: int,
) -> dict[str, object]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    duplicates = 0
    for source in sources:
        if not source.is_file():
            continue
        for row in _read_jsonl(source):
            fingerprint = _fingerprint(row)
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            row["source"] = str(source)
            row["fingerprint"] = fingerprint
            rows.append(row)

    rng = random.Random(seed)
    rng.shuffle(rows)
    validation_count = max(1, int(len(rows) * validation_ratio)) if rows else 0
    validation_rows = rows[:validation_count]
    train_rows = rows[validation_count:]

    output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with validation_output.open("w", encoding="utf-8") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    categories: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        categories[category] = categories.get(category, 0) + 1

    return {
        "total": len(rows),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "duplicates_removed": duplicates,
        "sources_used": [str(path) for path in sources if path.is_file()],
        "categories": categories,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble Indoone instruction data deterministically.")
    parser.add_argument("--source", action="append", type=Path, dest="sources")
    parser.add_argument("--output", type=Path, default=Path("data/processed/instructions_train.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("data/processed/instructions_validation.jsonl"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = args.sources or list(DEFAULT_SOURCES)
    report = assemble(sources, args.output, args.validation_output, args.validation_ratio, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
