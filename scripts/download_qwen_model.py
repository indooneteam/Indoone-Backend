from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Qwen/Qwen3-0.6B-GGUF"
DEFAULT_FILENAME = "Qwen3-0.6B-Q4_0.gguf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official Qwen3-0.6B GGUF candidate for local CPU inference."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--local-dir", type=Path, default=Path("models/qwen3-0.6b"))
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
    )
    print(downloaded)


if __name__ == "__main__":
    main()
