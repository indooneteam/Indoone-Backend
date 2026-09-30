from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.self_update import collect_and_update


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect trusted internet feed updates into Indoone reference knowledge."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("config/self_update_sources.json"),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/self_update/state.json"),
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("data/knowledge/auto_web.txt"),
    )
    args = parser.parse_args()

    report = collect_and_update(
        sources_path=args.sources,
        state_path=args.state,
        knowledge_path=args.knowledge,
    )
    print(
        json.dumps(
            {
                "changed": report.changed,
                "new_items": report.new_items,
                "retained_items": report.retained_items,
                "sources_checked": report.sources_checked,
                "source_errors": list(report.source_errors),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
