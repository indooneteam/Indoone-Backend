"""The parts of the live evaluation that can be tested without a model call.

The runner itself needs a provider key and spends money, so it is exercised by the
nightly job. What is tested here is the scoring: which cases run, and what counts as
a hit. Those are the parts where a mistake would quietly make the gate meaningless -
a scoring bug that counts a miss as a hit is worse than no gate at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.eval import tool_selection_eval as ev

GOLDEN_SET = Path(__file__).resolve().parents[3] / "data" / "eval" / "tool_selection.jsonl"


def _case(case_id: str = "c1", language: str = "en", expected: str = "github_list_branches") -> dict:
    return {"id": case_id, "language": language, "message": "m", "query": "q", "expected": expected}


# ---- loading and sampling -------------------------------------------------- #


def test_the_golden_set_loads_and_only_carries_cases_with_an_expected_tool() -> None:
    cases = ev.load_cases(GOLDEN_SET)

    assert len(cases) > 100
    assert all(str(case["expected"]).strip() for case in cases)


def test_a_missing_golden_set_is_an_error_not_an_empty_pass(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ev.load_cases(tmp_path / "absent.jsonl")


def test_the_sample_is_deterministic() -> None:
    cases = ev.load_cases(GOLDEN_SET)

    first = ev.sample_cases(cases, limit=20, seed=7)
    second = ev.sample_cases(cases, limit=20, seed=7)
    other = ev.sample_cases(cases, limit=20, seed=8)

    assert [case["id"] for case in first] == [case["id"] for case in second]
    assert [case["id"] for case in first] != [case["id"] for case in other]


def test_the_sample_covers_every_language_even_when_small() -> None:
    """A sample that is all English measures the language that already works."""

    cases = ev.load_cases(GOLDEN_SET)

    chosen = ev.sample_cases(cases, limit=25, seed=1)

    languages = {str(case["language"]) for case in chosen}
    assert {"en", "kn", "hi"} <= languages
    assert len(chosen) == 25


def test_a_limit_larger_than_the_set_returns_everything() -> None:
    cases = ev.load_cases(GOLDEN_SET)

    assert len(ev.sample_cases(cases, limit=10_000)) == len(cases)


# ---- scoring --------------------------------------------------------------- #


def test_calling_the_expected_tool_is_a_hit() -> None:
    result = ev.score_case(_case(), called=["github_list_branches"], discovered=[])

    assert result.outcome == ev.HIT_DIRECT
    assert result.hit is True


def test_reaching_it_through_find_tools_is_a_hit() -> None:
    """Discovery is how the loop is designed to work, so it is success, not a near miss."""

    result = ev.score_case(
        _case(), called=["find_tools", "github_list_branches"], discovered=["github_list_branches"]
    )

    assert result.outcome == ev.HIT_DIRECT


def test_find_tools_that_returns_the_tool_is_a_hit_even_if_the_second_round_never_ran() -> None:
    result = ev.score_case(_case(), called=["find_tools"], discovered=["github_list_branches"])

    assert result.outcome == ev.HIT_VIA_DISCOVERY
    assert result.hit is True


def test_answering_without_a_tool_is_a_miss() -> None:
    """This is the failure users report: the model answers from memory."""

    result = ev.score_case(_case(), called=[], discovered=[])

    assert result.outcome == ev.MISS_NO_TOOL
    assert result.hit is False


def test_calling_the_wrong_tool_is_a_miss() -> None:
    result = ev.score_case(_case(), called=["github_list_repositories"], discovered=[])

    assert result.outcome == ev.MISS_WRONG_TOOL
    assert "github_list_repositories" in result.detail


def test_find_tools_that_misses_is_its_own_outcome() -> None:
    """A discovery miss is a search problem, and must not look like a model problem."""

    result = ev.score_case(_case(), called=["find_tools"], discovered=["github_list_repositories"])

    assert result.outcome == ev.MISS_DISCOVERY


# ---- the verdict ----------------------------------------------------------- #


def _results(hits: int, misses: int, language: str = "en") -> list[ev.CaseResult]:
    return [ev.score_case(_case(f"h{i}", language), called=["github_list_branches"], discovered=[]) for i in range(hits)] + [
        ev.score_case(_case(f"m{i}", language), called=[], discovered=[]) for i in range(misses)
    ]


def test_the_report_passes_at_the_threshold() -> None:
    report = ev.summarise(_results(9, 1), threshold=0.90)

    assert report.accuracy == pytest.approx(0.9)
    assert report.passed is True


def test_the_report_fails_below_the_threshold() -> None:
    report = ev.summarise(_results(8, 2), threshold=0.90)

    assert report.accuracy == pytest.approx(0.8)
    assert report.passed is False
    assert len(report.failures) == 2


def test_an_empty_run_fails_rather_than_passing() -> None:
    """A gate that is green because nothing ran is worse than no gate."""

    report = ev.summarise([], threshold=0.9)

    assert report.passed is False
    assert report.total == 0


def test_the_report_breaks_accuracy_down_by_language() -> None:
    report = ev.summarise(_results(2, 0, "kn") + _results(1, 1, "hi"), threshold=0.5)

    assert report.by_language["kn"]["accuracy"] == 1.0
    assert report.by_language["hi"]["accuracy"] == 0.5
    assert report.by_outcome[ev.HIT_DIRECT] == 3
    assert report.by_outcome[ev.MISS_NO_TOOL] == 1


def test_the_report_serialises_for_the_artifact() -> None:
    payload = ev.summarise(_results(1, 1), threshold=0.9).as_dict()

    assert json.loads(json.dumps(payload))["total"] == 2
    assert payload["failures"][0]["id"] == "m0"
