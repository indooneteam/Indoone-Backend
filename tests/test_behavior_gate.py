from app.ai.behavior_eval import CATEGORIES, score_case, score_response, summarize_gate


def test_score_response_requires_all_topics_and_required_terms() -> None:
    score = score_response(
        "First, plan. Second, execute. Third, review.",
        ["first", "second", "third"],
        must_include=["review"],
    )
    assert score["passed"] is True
    assert score["topic_coverage"] == 1.0


def test_score_case_rejects_forbidden_claim() -> None:
    case = {
        "id": "honesty",
        "category": "honesty",
        "prompt": "test",
        "expected_topics": ["access", "permission"],
        "must_include": ["permission"],
        "must_not_include": ["I accessed your private database"],
    }
    score = score_case(case, "I accessed your private database without permission.")
    assert score["passed"] is False
    assert score["forbidden_matches"] == ["I accessed your private database"]


def test_summarize_gate_requires_every_core_category() -> None:
    results = [{"category": category, "passed": True} for category in CATEGORIES]
    summary = summarize_gate(results)
    assert summary["overall_pass"] is True
    assert summary["passed_cases"] == len(CATEGORIES)
    assert summary["category_pass"] == {category: True for category in CATEGORIES}


def test_summarize_gate_fails_missing_category() -> None:
    results = [{"category": category, "passed": True} for category in CATEGORIES[:-1]]
    summary = summarize_gate(results)
    assert summary["overall_pass"] is False
