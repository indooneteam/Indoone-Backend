from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.ai.qwen_runtime import QwenLocalModelRuntime


PROMPTS = (
    "Hello. Introduce yourself briefly.",
    "What can you help a user with? Give five short points.",
    "Namaskara. Nimage yava kelasa madoke agutte?",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    runtime = QwenLocalModelRuntime(
        args.model,
        context_size=1024,
        threads=2,
    )

    for prompt in PROMPTS:
        started = time.perf_counter()
        reply = runtime.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
        )
        elapsed = time.perf_counter() - started
        if not reply.strip():
            raise SystemExit(f"Qwen returned an empty reply for: {prompt}")
        print(f"PROMPT: {prompt}")
        print(f"SECONDS: {elapsed:.2f}")
        print(f"REPLY: {reply}")
        print("---")


if __name__ == "__main__":
    main()
