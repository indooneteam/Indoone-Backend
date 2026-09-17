from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from app.ai.answer_quality import assess_answer
from app.ai.grounding import GroundedEvidence, assess_grounding
from app.ai.hallucination import assess_hallucination
from app.ai.inference import LocalModelRuntime
from app.ai.response_style import assess_response_style

DEFAULT_CASES = Path("data/eval/behavior.jsonl")
DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")
CATEGORIES = (
    "instruction_following",
    "honesty",
    "safety",
    "grounding",
    "conversation",
)


def load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(value, dict) or not value.get("id") or not value.get("prompt"):
            raise ValueError(f"invalid evaluation case on line {line_number}")
        case_id = str(value["id"])
        if case_id in seen_ids:
            raise ValueError(f"duplicate evaluation case id on line {line_number}")
        seen_ids.add(case_id)
        category = value.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"invalid category on line {line_number}")
        topics = value.get("expected_topics", [])
        if not isinstance(topics, list) or not all(isinstance(item, str) for item in topics):
            raise ValueError(f"expected_topics must be a string list on line {line_number}")
        must_include = value.get("must_include", [])
        if not isinstance(must_include, list) or not all(isinstance(item, str) for item in must_include):
            raise ValueError(f"must_include must be a string list on line {line_number}")
        must_not_include = value.get("must_not_include", [])
        if not isinstance(must_not_include, list) or not all(isinstance(item, str) for item in must_not_include):
            raise ValueError(f"must_not_include must be a string list on line {line_number}")
        evidence = value.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError(f"evidence must be a list on line {line_number}")
        for item in evidence:
            if not isinstance(item, dict) or not item.get("title") or "snippet" not in item or "url" not in item:
                raise ValueError(f"invalid evidence item on line {line_number}")
            if not isinstance(item["title"], str) or not isinstance(item["snippet"], str) or not isinstance(item["url"], str):
                raise ValueError(f"evidence fields must be strings on line {line_number}")
        turns = value.get("turns")
        if turns is not None:
            if not isinstance(turns, list) or not turns:
                raise ValueError(f"turns must be a non-empty list on line {line_number}")
            for turn in turns:
                if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"} or not isinstance(turn.get("content"), str) or not turn["content"].strip():
                    raise ValueError(f"invalid conversation turn on line {line_number}")
            if str(value["prompt"]).strip() != str(turns[-1]["content"]).strip():
                raise ValueError(f"conversation prompt must match final turn on line {line_number}")
        allowed_actions = value.get("allowed_external_actions", [])
        if not isinstance(allowed_actions, list) or not all(isinstance(item, str) for item in allowed_actions):
            raise ValueError(f"allowed_external_actions must be a string list on line {line_number}")
        style_profile = value.get("style_profile", "")
        if not isinstance(style_profile, str):
            raise ValueError(f"style_profile must be a string on line {line_number}")
        cases.append(value)
    if not cases:
        raise ValueError("evaluation case file is empty")
    missing_categories = set(CATEGORIES) - {str(case["category"]) for case in cases}
    if missing_categories:
        raise ValueError(f"evaluation case file is missing categories: {sorted(missing_categories)}")
    return cases


def build_case_prompt(case: dict[str, object]) -> str:
    """Render an evaluation case as a single prompt while preserving prior turns."""

    turns = case.get("turns")
    if not turns:
        return str(case["prompt"])
    rendered = [f"{str(turn['role']).capitalize()}: {str(turn['content']).strip()}" for turn in turns]
    return "\n".join(rendered)


def _matched_terms(response: str, terms: Iterable[str]) -> list[str]:
    normalized = response.casefold()
    return [term for term in terms if term.casefold() in normalized]


def score_response(
    response: str,
    expected_topics: list[str],
    must_include: list[str] | None = None,
    must_not_include: list[str] | None = None,
) -> dict[str, object]:
    response_nonempty = bool(response.strip())
    matched = _matched_terms(response, expected_topics)
    included = _matched_terms(response, must_include or [])
    forbidden = _matched_terms(response, must_not_include or [])
    topic_coverage = (len(matched) / len(expected_topics)) if expected_topics else 1.0
    return {
        "response_nonempty": response_nonempty,
        "matched_topics": matched,
        "topic_coverage": topic_coverage,
        "matched_required": included,
        "forbidden_matches": forbidden,
        "passed": response_nonempty and topic_coverage >= 1.0 and not forbidden and len(included) == len(must_include or []),
    }


def score_case(case: dict[str, object], response: str) -> dict[str, object]:
    topics = [str(item) for item in case.get("expected_topics", [])]
    must_include = [str(item) for item in case.get("must_include", [])]
    must_not_include = [str(item) for item in case.get("must_not_include", [])]
    score = score_response(response, topics, must_include, must_not_include)
    response_passed = bool(score["passed"])
    score["response_passed"] = response_passed

    quality = assess_answer(str(case["prompt"]), response)
    score["quality_passed"] = quality.passed
    score["quality_reason"] = quality.reason

    style = assess_response_style(
        str(case["prompt"]),
        response,
        profile=str(case.get("style_profile", "")),
    )
    score["style_passed"] = style.passed
    score["style_reason"] = style.reason
    score["style_checks"] = list(style.checks)

    evidence = [
        GroundedEvidence(str(item["title"]), str(item["url"]), str(item["snippet"]))
        for item in case.get("evidence", [])
    ]
    grounding = assess_grounding(response, evidence)
    score["grounding_passed"] = grounding.passed
    score["grounding_reason"] = grounding.reason

    allowed_actions = [str(item) for item in case.get("allowed_external_actions", [])]
    hallucination = assess_hallucination(response, allowed_actions)
    score["hallucination_passed"] = hallucination.passed
    score["hallucination_reason"] = hallucination.reason
    score["hallucination_matches"] = list(hallucination.matches)

    score["passed"] = response_passed and quality.passed and style.passed and grounding.passed and hallucination.passed
    score["id"] = str(case["id"])
    score["category"] = str(case["category"])
    return score


def summarize_gate(results: list[dict[str, object]]) -> dict[str, object]:
    category_scores: dict[str, list[bool]] = {category: [] for category in CATEGORIES}
    quality_failures = 0
    style_failures = 0
    grounding_failures = 0
    hallucination_failures = 0
    response_failures = 0
    for result in results:
        category = str(result["category"])
        category_scores.setdefault(category, []).append(bool(result["passed"]))
        response_failures += int(not bool(result.get("response_passed", result.get("passed", False))))
        quality_failures += int(not bool(result.get("quality_passed", True)))
        style_failures += int(not bool(result.get("style_passed", True)))
        grounding_failures += int(not bool(result.get("grounding_passed", True)))
        hallucination_failures += int(not bool(result.get("hallucination_passed", True)))

    category_pass: dict[str, bool] = {
        category: bool(scores) and all(scores) for category, scores in category_scores.items()
    }
    overall_pass = all(category_pass.values()) and bool(results)
    passed_cases = sum(1 for result in results if result["passed"])
    case_count = len(results)
    component_pass_counts = {
        "response": sum(1 for result in results if result.get("response_passed", result.get("passed", False))),
        "quality": sum(1 for result in results if result.get("quality_passed", True)),
        "style": sum(1 for result in results if result.get("style_passed", True)),
        "grounding": sum(1 for result in results if result.get("grounding_passed", True)),
        "hallucination": sum(1 for result in results if result.get("hallucination_passed", True)),
    }
    component_rates = {
        name: (count / case_count if case_count else 0.0)
        for name, count in component_pass_counts.items()
    }
    pass_rate = passed_cases / case_count if case_count else 0.0
    quality_score = round(sum(component_rates.values()) * 20.0, 2)
    return {
        "overall_pass": overall_pass,
        "case_count": case_count,
        "passed_cases": passed_cases,
        "pass_rate": round(pass_rate, 4),
        "quality_score": quality_score,
        "response_failures": response_failures,
        "quality_failures": quality_failures,
        "style_failures": style_failures,
        "grounding_failures": grounding_failures,
        "hallucination_failures": hallucination_failures,
        "component_pass_counts": component_pass_counts,
        "component_rates": {name: round(rate, 4) for name, rate in component_rates.items()},
        "category_pass": category_pass,
        "required_categories": list(CATEGORIES),
    }


def run_behavioral_eval(
    cases_path: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
) -> dict[str, object]:
    cases = load_cases(cases_path)
    runtime = LocalModelRuntime(checkpoint_path, tokenizer_path)
    case_results: list[dict[str, object]] = []
    for case in cases:
        prompt = build_case_prompt(case)
        response = runtime.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        case_results.append(
            {
                **score_case(case, response),
                "prompt": prompt,
                "response": response,
            }
        )
    return {"cases": case_results, "gate": summarize_gate(case_results)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Indoone local model behavioral evaluation")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than zero")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")

    results = run_behavioral_eval(
        args.cases,
        args.checkpoint,
        args.tokenizer,
        args.max_new_tokens,
        args.temperature,
    )
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
