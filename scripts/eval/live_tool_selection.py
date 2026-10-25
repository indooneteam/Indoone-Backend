"""Run the golden set against the live model and gate on the result.

What this measures, and why it is not the pull-request gate
-----------------------------------------------------------
`tests/ai/execution/test_tool_selection_golden.py` measures the search: given the
English query, is the right tool in the shortlist. That runs on every pull request
because it is free and fast, and it is necessary - but it is not sufficient. It
cannot see the failure that actually reaches users:

    the model answers from its own memory instead of calling a tool at all.

Nothing in the search can fix that, because the search is never reached. So this
script runs a sample of the same golden set against the real provider, records what
the model chose to call, and fails when the accuracy drops below a threshold.

It runs nightly rather than per pull request because every case is a paid model call.

Usage
-----
    python scripts/eval/live_tool_selection.py
    python scripts/eval/live_tool_selection.py --sample 60 --threshold 0.92
    python scripts/eval/live_tool_selection.py --report eval-report.json

Environment
-----------
    INDOONE_EVAL_SAMPLE      cases to run (default 40)
    INDOONE_EVAL_THRESHOLD   accuracy required to pass (default 0.90)
    INDOONE_EVAL_SEED        sample seed, so a failed night reproduces (default 0)
    GEMINI_API_KEY           the key the provider call needs

Exit code is 0 when the threshold is met and 1 when it is not, so a scheduled job
goes red without any extra wiring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Importing the application is the point: the declarations, the system instruction and
# the discovery search must be the ones production uses, or the number measured here
# says nothing about production.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ai.eval.tool_selection_eval import (  # noqa: E402
    CaseResult,
    load_cases,
    sample_cases,
    score_case,
    summarise,
)
from app.ai.execution.tool_loop import model_tool_declarations  # noqa: E402
from app.ai.execution.tool_search import find_tools_payload  # noqa: E402

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "data" / "eval" / "tool_selection.jsonl"
FIND_TOOLS_LIMIT = 5


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0.0 < value <= 1.0 else default


async def _run_case(case: dict[str, Any]) -> CaseResult:
    """One case: give the model the same tools production gives it, and watch.

    ``execute_tool`` never touches a connector. It records what was asked for, and for
    ``find_tools`` it runs the real search so a model that discovers the tool is scored
    as having reached it - which is how the loop is designed to work.
    """

    from app.ai.gateway.gemini_service import generate_gemini_function_call_reply

    called: list[str] = []
    discovered: set[str] = set()

    async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        called.append(name)
        if name == "find_tools":
            payload = find_tools_payload(str(arguments.get("query") or ""), limit=FIND_TOOLS_LIMIT)
            if payload.get("ok") is True:
                names = [
                    str(item["name"])
                    for item in (payload.get("result") or {}).get("tools") or []
                    if isinstance(item, dict) and item.get("name")
                ]
                discovered.update(names)
                return payload
        # Any other tool: report success without doing anything. This measures the
        # decision to call, not the connector behind it.
        return {"ok": True, "result": {"evaluated": True}}

    def declarations() -> list[dict[str, Any]]:
        return model_tool_declarations("", discovered=tuple(discovered))

    message = str(case.get("query") or case.get("message") or "").strip()
    try:
        await generate_gemini_function_call_reply(
            message,
            history=None,
            tool_declarations=declarations,
            execute_tool=execute_tool,
            extra_system_instruction="",
            # Two rounds is enough to see "call find_tools, then call the tool", and it
            # keeps a nightly run affordable.
            max_tool_rounds=2,
            tool_mode="AUTO",
            return_none_when_no_tool=True,
        )
    except Exception as exc:  # noqa: BLE001 - a provider failure is a failed case, not a crash
        return CaseResult(
            case_id=str(case.get("id") or ""),
            language=str(case.get("language") or ""),
            expected=str(case.get("expected") or ""),
            outcome="miss_no_tool",
            called=tuple(called),
            discovered=tuple(sorted(discovered)),
            detail=f"provider error: {type(exc).__name__}: {exc}",
        )

    return score_case(case, called=called, discovered=discovered)


async def _run(cases: list[dict[str, Any]]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        result = await _run_case(case)
        results.append(result)
        mark = "ok  " if result.hit else "MISS"
        print(f"[{index}/{len(cases)}] {mark} {result.case_id} {result.language} -> {result.detail}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live-model tool-selection evaluation.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="path to the golden set")
    parser.add_argument("--sample", type=int, default=_env_int("INDOONE_EVAL_SAMPLE", 40))
    parser.add_argument("--threshold", type=float, default=_env_float("INDOONE_EVAL_THRESHOLD", 0.90))
    parser.add_argument("--seed", type=int, default=_env_int("INDOONE_EVAL_SEED", 0))
    parser.add_argument("--report", default=os.getenv("INDOONE_EVAL_REPORT", "").strip())
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    chosen = sample_cases(cases, limit=args.sample, seed=args.seed)
    print(f"running {len(chosen)} of {len(cases)} cases, threshold {args.threshold:.0%}", flush=True)

    results = asyncio.run(_run(chosen))
    report = summarise(results, threshold=args.threshold)
    payload = report.as_dict()
    payload["sample"] = len(chosen)
    payload["available"] = len(cases)
    payload["seed"] = args.seed

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.report:
        Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.report}", flush=True)

    if not report.passed:
        print(
            f"FAILED: tool-selection accuracy {report.accuracy:.1%} is below the "
            f"{report.threshold:.0%} threshold",
            file=sys.stderr,
        )
        return 1
    print(f"PASSED: tool-selection accuracy {report.accuracy:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
