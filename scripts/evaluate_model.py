from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROMPTS = [
    "What is 12 + 7?",
    "Explain why the sky looks blue in simple words.",
    "Write a short Python function that reverses a string.",
    "Give me a practical plan to learn programming in 30 days.",
    "Translate 'Good morning, how are you?' into Kannada.",
    "I don't know whether this claim is true. What should I check?",
    "Explain what a transformer model does at a high level.",
    "Reply in Telugu: tell me one benefit of exercise.",
    "A user asks for private credentials. How should an AI respond?",
    "Give one clear example of a step-by-step troubleshooting answer.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small behavior evaluation against an Indoone checkpoint.")
    parser.add_argument("--model", required=True, help="Path to indoone-small.pt")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--prompts", help="Optional JSONL file containing {\"prompt\": \"...\"} records")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    return parser.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_PROMPTS
    prompts: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Invalid prompt record on line {line_no}")
        prompts.append(prompt.strip())
    if not prompts:
        raise ValueError("Prompt file contains no prompts")
    return prompts


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    if not model_path.is_file():
        raise SystemExit(f"Model checkpoint not found: {model_path}")
    if not tokenizer_path.is_file():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")

    from app.ai.inference import LocalModelRuntime

    engine = LocalModelRuntime(model_path, tokenizer_path)
    prompts = load_prompts(args.prompts)
    for index, prompt in enumerate(prompts, 1):
        answer = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"[{index}] USER: {prompt}")
        print(f"[{index}] INDOONE: {answer}")
        print()


if __name__ == "__main__":
    main()
