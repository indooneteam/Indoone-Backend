from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_ITEMS = 120


def _load_items(state_path: Path) -> list[dict[str, str]]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("self-update state must contain an items list")

    items: list[dict[str, str]] = []
    for raw in payload["items"]:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source", "")).strip()
        title = str(raw.get("title", "")).strip()
        url = str(raw.get("url", "")).strip()
        summary = str(raw.get("summary", "")).strip()
        published = str(raw.get("published", "")).strip()
        fingerprint = str(raw.get("fingerprint", "")).strip()
        if source and title and url and summary and fingerprint:
            items.append(
                {
                    "source": source,
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "published": published,
                    "fingerprint": fingerprint,
                }
            )
    return items[:MAX_ITEMS]


def build_examples(state_path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for item in _load_items(state_path):
        citation = f"Source: {item['source']}\nURL: {item['url']}"
        if item["published"]:
            citation += f"\nPublished: {item['published']}"

        response = f"{item['summary']}\n{citation}"
        examples.extend(
            [
                {
                    "instruction": f"Summarize the latest update titled '{item['title']}'.",
                    "response": response,
                    "category": "web_update",
                    "fingerprint": item["fingerprint"],
                },
                {
                    "instruction": f"What did {item['source']} report about '{item['title']}'?",
                    "response": response,
                    "category": "web_update",
                    "fingerprint": item["fingerprint"],
                },
                {
                    "instruction": f"Give the key facts from the update '{item['title']}'.",
                    "response": response,
                    "category": "web_update",
                    "fingerprint": item["fingerprint"],
                },
            ]
        )
    return examples


def merge_into_training_file(
    *,
    state_path: Path,
    output_path: Path,
) -> int:
    new_examples = build_examples(state_path)

    existing: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    if output_path.is_file():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            instruction = str(payload.get("instruction", "")).strip()
            response = str(payload.get("response", "")).strip()
            if not instruction or not response:
                continue
            key = (instruction.casefold(), response.casefold())
            if key in seen:
                continue
            seen.add(key)
            existing.append(payload)

    added = 0
    for example in new_examples:
        key = (
            example["instruction"].casefold(),
            example["response"].casefold(),
        )
        if key in seen:
            continue
        existing.append(example)
        seen.add(key)
        added += 1

    if added:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered = "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in existing
        )
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return added


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge trusted web knowledge into the existing generated training pool."
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/self_update/state.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/generated_multilingual_examples.jsonl"),
    )
    args = parser.parse_args()

    added = merge_into_training_file(
        state_path=args.state,
        output_path=args.output,
    )
    print(json.dumps({"added_examples": added}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
