from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.ai.gguf_runtime import GGUFModelRuntime


PROMPTS = (
    "Hello. Introduce yourself briefly.",
    "What can you help a user with? Give five short points.",
    "ನಮಸ್ಕಾರ. ನೀವು ಏನು ಮಾಡಬಹುದು?",
    "Namaskara. Nimage enu madoke agutte?",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    runtime = GGUFModelRuntime(args.model, context_size=512, threads=2)

    for prompt in PROMPTS:
        started = time.perf_counter()
        reply = runtime.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )
        elapsed = time.perf_counter() - started
        if not reply.strip():
            raise SystemExit(f"Empty reply for: {prompt}")
        print(f"PROMPT: {prompt}")
        print(f"SECONDS: {elapsed:.2f}")
        print(f"REPLY: {reply}")
        print("---")


if __name__ == "__main__":
    main()
