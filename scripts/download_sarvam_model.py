from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "bartowski/sarvam-1-GGUF"
DEFAULT_FILENAME = "sarvam-1-Q4_K_M.gguf"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the Sarvam-1 Q4_K_M GGUF candidate."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--local-dir", type=Path, default=Path("models/sarvam-1"))
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=str(args.local_dir),
        )
    )

    actual = _sha256(downloaded)
    if args.expected_sha256 and actual.casefold() != args.expected_sha256.casefold():
        raise SystemExit(
            f"SHA-256 mismatch for {downloaded}: expected {args.expected_sha256}, got {actual}"
        )

    print(f"Verified Sarvam model: {downloaded}")
    print(f"SHA-256: {actual}")


if __name__ == "__main__":
    main()
