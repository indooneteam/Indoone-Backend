from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.storage.b2 import B2Storage, B2StorageError


DEFAULT_FILENAMES = (
    "indoone-small.pt",
    "tokenizer.json",
    "metadata.json",
    "training_history.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload trained Indoone artifacts to Backblaze B2."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/indoone-small"),
    )
    parser.add_argument(
        "--prefix",
        default="models/indoone-small",
        help="B2 object prefix for the uploaded artifacts.",
    )
    args = parser.parse_args()

    try:
        storage = B2Storage()
        storage.check_access()
    except B2StorageError as exc:
        raise SystemExit(str(exc)) from exc

    missing = [
        name for name in DEFAULT_FILENAMES if not (args.model_dir / name).is_file()
    ]
    if missing:
        raise SystemExit(
            "Missing trained artifacts: " + ", ".join(missing)
        )

    for filename in DEFAULT_FILENAMES:
        storage.upload_file(
            args.model_dir / filename,
            f"{args.prefix.rstrip('/')}/{filename}",
        )
        print(f"uploaded {filename}")


if __name__ == "__main__":
    main()
