from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


DEFAULT_MIN_CORPUS_CHARS = 1_000_000
DEFAULT_MIN_INSTRUCTION_EXAMPLES = 10_000
DEFAULT_MIN_MULTILINGUAL_EXAMPLES = 1_000
DEFAULT_MAX_DUPLICATE_RATE = 0.05


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _pair_key(row: dict[str, object]) -> tuple[str, str]:
    instruction = str(row.get("instruction", "")).strip().casefold()
    response = str(row.get("response", "")).strip().casefold()
    return instruction, response


def _language_bucket(text: str) -> str:
    counts = Counter()
    for char in text:
        code = ord(char)
        if 0x0C80 <= code <= 0x0CFF:
            counts["kn"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["hi"] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            counts["te"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["ta"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            counts["ml"] += 1
        elif 0x0980 <= code <= 0x09FF:
            counts["bn"] += 1
        elif 0x0A80 <= code <= 0x0AFF:
            counts["gu"] += 1
        elif 0x0A00 <= code <= 0x0A7F:
            counts["pa"] += 1
        elif 0x0B00 <= code <= 0x0B7F:
            counts["or"] += 1
        elif 0x0600 <= code <= 0x06FF:
            counts["ur"] += 1
        elif char.isascii() and char.isalpha():
            counts["latin"] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def audit_dataset(
    corpus_path: Path,
    instruction_paths: list[Path],
    multilingual_path: Path | None,
    min_corpus_chars: int = DEFAULT_MIN_CORPUS_CHARS,
    min_instruction_examples: int = DEFAULT_MIN_INSTRUCTION_EXAMPLES,
    min_multilingual_examples: int = DEFAULT_MIN_MULTILINGUAL_EXAMPLES,
    max_duplicate_rate: float = DEFAULT_MAX_DUPLICATE_RATE,
) -> dict[str, object]:
    if not 0 <= max_duplicate_rate <= 1:
        raise ValueError("max_duplicate_rate must be between 0 and 1")

    corpus = corpus_path.read_text(encoding="utf-8") if corpus_path.exists() else ""
    rows: list[dict[str, object]] = []
    for path in instruction_paths:
        rows.extend(_read_jsonl(path))

    keys = [_pair_key(row) for row in rows]
    non_empty_keys = [key for key in keys if key != ("", "")]
    unique_keys = set(non_empty_keys)
    duplicate_rate = (
        1 - (len(unique_keys) / len(non_empty_keys)) if non_empty_keys else 1.0
    )

    categories = Counter(str(row.get("category", "uncategorized")) for row in rows)
    languages = Counter(
        _language_bucket(f"{row.get('instruction', '')} {row.get('response', '')}")
        for row in rows
    )

    multilingual_count = sum(
        1
        for row in (_read_jsonl(multilingual_path) if multilingual_path else [])
        if _pair_key(row) != ("", "")
    )

    failures: list[str] = []
    if len(corpus) < min_corpus_chars:
        failures.append(
            f"corpus chars {len(corpus)} < required {min_corpus_chars}"
        )
    if len(rows) < min_instruction_examples:
        failures.append(
            f"instruction examples {len(rows)} < required {min_instruction_examples}"
        )
    if multilingual_count < min_multilingual_examples:
        failures.append(
            f"multilingual examples {multilingual_count} < required {min_multilingual_examples}"
        )
    if duplicate_rate > max_duplicate_rate:
        failures.append(
            f"duplicate rate {duplicate_rate:.2%} > allowed {max_duplicate_rate:.2%}"
        )

    return {
        "ready": not failures,
        "corpus_path": str(corpus_path),
        "corpus_characters": len(corpus),
        "instruction_examples": len(rows),
        "multilingual_examples": multilingual_count,
        "unique_instruction_pairs": len(unique_keys),
        "duplicate_rate": duplicate_rate,
        "categories": dict(categories),
        "language_buckets": dict(languages),
        "failures": failures,
        "requirements": {
            "min_corpus_chars": min_corpus_chars,
            "min_instruction_examples": min_instruction_examples,
            "min_multilingual_examples": min_multilingual_examples,
            "max_duplicate_rate": max_duplicate_rate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether Indoone training data is ready for a serious model run")
    parser.add_argument("--corpus", type=Path, default=Path("data/raw/indoone_corpus.txt"))
    parser.add_argument(
        "--instructions",
        type=Path,
        nargs="+",
        default=[Path("data/raw/indoone_instructions.jsonl")],
    )
    parser.add_argument(
        "--multilingual",
        type=Path,
        default=Path("data/raw/indoone_multilingual_examples.jsonl"),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    report = audit_dataset(args.corpus, args.instructions, args.multilingual)
    if args.as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"dataset_ready={report['ready']}")
        print(f"corpus_characters={report['corpus_characters']}")
        print(f"instruction_examples={report['instruction_examples']}")
        print(f"multilingual_examples={report['multilingual_examples']}")
        print(f"duplicate_rate={report['duplicate_rate']:.2%}")
        if report["failures"]:
            print("blocking_findings:")
            for failure in report["failures"]:
                print(f"- {failure}")

    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
