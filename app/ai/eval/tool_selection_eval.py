"""Scoring for the live-model tool-selection evaluation.

The CI gate in `tests/ai/execution/test_tool_selection_golden.py` measures the
*search*: given the English query, is the right tool in the shortlist. That gate is
fast, free and runs on every pull request, and it cannot answer the question that
actually decides whether the assistant works:

    does the model choose to call a tool at all?

A model that answers from memory, or that reads its own old "I don't have access"
and repeats it, never reaches the search. No amount of search quality fixes that. So
it is measured against the live model, on a schedule rather than on every pull
request, and this module holds the part that can be tested without a network call:
which cases to run, and how to score what came back.

The scoring is deliberately generous in one specific way and strict in another.

* Generous: a case counts as a hit if the model called the expected tool directly,
  **or** if it called `find_tools` and the tool the search returned was the expected
  one. Discovery is how the loop is designed to work, so reaching the tool through it
  is success, not a near miss.
* Strict: a case where the model called a different tool is a miss, and a case where
  it called nothing is a miss. Those are the two failures that reach a user.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# The outcomes a case can end in. Named rather than booleans so a report says what
# went wrong instead of only that something did.
HIT_DIRECT = "hit_direct"
HIT_VIA_DISCOVERY = "hit_via_discovery"
MISS_NO_TOOL = "miss_no_tool"
MISS_WRONG_TOOL = "miss_wrong_tool"
MISS_DISCOVERY = "miss_discovery"

HITS = frozenset({HIT_DIRECT, HIT_VIA_DISCOVERY})

# Every language the golden set covers. Sampling keeps this coverage even when only a
# few dozen cases are run, because the whole point of the eval is the languages where
# the model is most likely to answer without calling anything.
LANGUAGES = ("en", "kn", "hi", "ta", "mixed")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    language: str
    expected: str
    outcome: str
    called: tuple[str, ...] = ()
    discovered: tuple[str, ...] = ()
    detail: str = ""

    @property
    def hit(self) -> bool:
        return self.outcome in HITS


@dataclass(frozen=True)
class Report:
    total: int
    hits: int
    accuracy: float
    threshold: float
    passed: bool
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    by_outcome: dict[str, int] = field(default_factory=dict)
    failures: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "hits": self.hits,
            "accuracy": round(self.accuracy, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "by_language": self.by_language,
            "by_outcome": self.by_outcome,
            "failures": list(self.failures),
        }


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """The golden set, as a list. Raises if the file is missing or malformed."""

    location = Path(path)
    if not location.is_file():
        raise FileNotFoundError(f"golden set not found: {location}")
    cases: list[dict[str, Any]] = []
    for line in location.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            if str(case.get("expected") or "").strip():
                cases.append(case)
    return cases


def sample_cases(
    cases: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """A deterministic, language-balanced sample.

    Round-robin over languages rather than a straight slice: a straight sample of
    thirty cases from this set is mostly English, and English is the language where
    tool selection already works. The interesting cases are the Kannada, Hindi, Tamil
    and mixed ones, so every language is taken from in turn until the limit is met.

    The same seed always returns the same sample, so a night that fails can be
    reproduced exactly.
    """

    if limit <= 0 or limit >= len(cases):
        return list(cases)

    random.Random(seed).shuffle(cases_list := list(cases))
    by_language: dict[str, list[dict[str, Any]]] = {language: [] for language in LANGUAGES}
    other: list[dict[str, Any]] = []
    for case in cases_list:
        language = str(case.get("language") or "").strip()
        if language in by_language:
            by_language[language].append(case)
        else:
            other.append(case)

    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit:
        progressed = False
        for language in LANGUAGES:
            bucket = by_language[language]
            if bucket and len(chosen) < limit:
                chosen.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break
    for case in other:
        if len(chosen) >= limit:
            break
        chosen.append(case)
    return chosen


def score_case(
    case: dict[str, Any],
    *,
    called: Iterable[str],
    discovered: Iterable[str],
) -> CaseResult:
    """Score one case against what the model actually did.

    ``called`` is every tool the model asked for, in order. ``discovered`` is what
    ``find_tools`` returned for the queries it asked, already flattened.
    """

    expected = str(case.get("expected") or "").strip()
    names = tuple(str(name) for name in called)
    found = tuple(str(name) for name in discovered)
    case_id = str(case.get("id") or "")
    language = str(case.get("language") or "")

    if expected in names:
        outcome, detail = HIT_DIRECT, "called the expected tool"
    elif expected in found:
        outcome, detail = HIT_VIA_DISCOVERY, "found it through find_tools"
    elif not names:
        outcome, detail = MISS_NO_TOOL, "answered without calling any tool"
    elif "find_tools" in names:
        outcome, detail = MISS_DISCOVERY, "find_tools did not return the expected tool"
    else:
        outcome, detail = MISS_WRONG_TOOL, f"called {', '.join(names)}"

    return CaseResult(
        case_id=case_id,
        language=language,
        expected=expected,
        outcome=outcome,
        called=names,
        discovered=found,
        detail=detail,
    )


def summarise(results: Sequence[CaseResult], *, threshold: float) -> Report:
    """Accuracy overall and per language, and whether it cleared the threshold."""

    total = len(results)
    hits = sum(1 for result in results if result.hit)
    accuracy = (hits / total) if total else 0.0

    by_language: dict[str, dict[str, float]] = {}
    for result in results:
        bucket = by_language.setdefault(result.language or "unknown", {"total": 0, "hits": 0, "accuracy": 0.0})
        bucket["total"] += 1
        if result.hit:
            bucket["hits"] += 1
    for bucket in by_language.values():
        bucket["accuracy"] = round(bucket["hits"] / bucket["total"], 4) if bucket["total"] else 0.0

    by_outcome: dict[str, int] = {}
    for result in results:
        by_outcome[result.outcome] = by_outcome.get(result.outcome, 0) + 1

    failures = tuple(
        {
            "id": result.case_id,
            "language": result.language,
            "expected": result.expected,
            "outcome": result.outcome,
            "called": list(result.called),
            "detail": result.detail,
        }
        for result in results
        if not result.hit
    )

    return Report(
        total=total,
        hits=hits,
        accuracy=accuracy,
        threshold=threshold,
        # An empty run is a failed run: a gate that passes because nothing ran is worse
        # than no gate, because it looks green.
        passed=bool(total) and accuracy >= threshold,
        by_language=by_language,
        by_outcome=by_outcome,
        failures=failures,
    )
