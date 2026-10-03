from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.ai.sarvam_runtime import SarvamLocalModelRuntime


PROMPTS = (
    "Hello. Introduce yourself briefly.",
    "What can you help a user with? Give five short points.",
    "Namaskara. Nimage yava kelasa madoke agutte?",
    "Ondu simple Kannada sentence nalli, Bengaluru bagge heli.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    runtime = SarvamLocalModelRuntime(
        args.model,
        context_size=2048,
        threads=2,
    )

    for prompt in PROMPTS:
        started = time.perf_counter()
        reply = runtime.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Indoone AI, a helpful assistant. "
                        "Answer the user in the same language they use. "
                        "Do not repeat the user's question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=128,
            temperature=0.2,
            top_p=0.9,
        )
        elapsed = time.perf_counter() - started
        if not reply.strip():
            raise SystemExit(f"Sarvam returned an empty reply for: {prompt}")
        print(f"PROMPT: {prompt}")
        print(f"SECONDS: {elapsed:.2f}")
        print(f"REPLY: {reply}")
        print("---")


if __name__ == "__main__":
    main()
