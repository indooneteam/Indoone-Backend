from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "bartowski/google_gemma-3-270m-it-GGUF"
DEFAULT_FILENAME = "google_gemma-3-270m-it-Q4_K_M.gguf"
DEFAULT_SHA256 = "c866c9f113f2e9aa2225c5997ede437392b8fa844ba5db9e4c77e315ffe20800"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--expected-sha256", default=DEFAULT_SHA256)
    parser.add_argument("--local-dir", type=Path, default=Path("models/gemma3-270m-it"))
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(args.local_dir),
        )
    )
    actual = sha256(path)
    if actual.casefold() != args.expected_sha256.casefold():
        raise SystemExit(
            f"SHA-256 mismatch: expected {args.expected_sha256}, got {actual}"
        )
    print(f"Verified: {path}")
    print(f"SHA-256: {actual}")


if __name__ == "__main__":
    main()
