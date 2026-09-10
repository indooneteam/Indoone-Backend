from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ai.inference import LocalModelRuntime

DEFAULT_CASES = Path("data/eval/behavior.jsonl")
DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")


def load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(value, dict) or not value.get("id") or not value.get("prompt"):
            raise ValueError(f"invalid evaluation case on line {line_number}")
        topics = value.get("expected_topics", [])
        if not isinstance(topics, list) or not all(isinstance(item, str) for item in topics):
            raise ValueError(f"expected_topics must be a string list on line {line_number}")
        cases.append(value)
    if not cases:
        raise ValueError("evaluation case file is empty")
    return cases


def score_response(response: str, expected_topics: list[str]) -> dict[str, object]:
    normalized = response.casefold()
    matched = [topic for topic in expected_topics if topic.casefold() in normalized]
    return {
        "response_nonempty": bool(response.strip()),
        "matched_topics": matched,
        "topic_coverage": (len(matched) / len(expected_topics)) if expected_topics else 1.0,
    }


def run_behavioral_eval(
    cases_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> list[dict[str, object]]:
    cases = load_cases(cases_path)
    runtime = LocalModelRuntime(checkpoint_path, tokenizer_path)
    results: list[dict[str, object]] = []
    for case in cases:
        prompt = str(case["prompt"])
        topics = [str(item) for item in case.get("expected_topics", [])]
        response = runtime.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        results.append(
            {
                "id": str(case["id"]),
                "prompt": prompt,
                "response": response,
                "score": score_response(response, topics),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Indoone local model behavioral evaluation")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")

    results = run_behavioral_eval(
        args.cases,
        args.checkpoint,
        args.tokenizer,
        args.max_new_tokens,
        args.temperature,
    )
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
