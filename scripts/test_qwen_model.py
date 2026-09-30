from __future__ import annotations

import argparse
from pathlib import Path

from app.ai.qwen_runtime import QwenLocalModelRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    runtime = QwenLocalModelRuntime(args.model)
    reply = runtime.generate(
        [{"role": "user", "content": "Hello. Introduce yourself briefly."}],
        max_tokens=64,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
    )
    print(reply)


if __name__ == "__main__":
    main()
