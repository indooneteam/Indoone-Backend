from app.ai.behavior_eval import CATEGORIES, build_case_prompt, score_case, score_response, summarize_gate


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


def test_score_case_applies_freshness_quality_gate() -> None:
    case = {
        "id": "freshness",
        "category": "grounding",
        "prompt": "What is the latest Indoone release?",
        "expected_topics": ["latest"],
    }
    score = score_case(case, "The latest Indoone release is version 2.")
    assert score["passed"] is False
    assert score["quality_passed"] is False
    assert score["quality_reason"] == "freshness_not_supported"


def test_score_case_accepts_freshness_with_sources() -> None:
    case = {
        "id": "freshness",
        "category": "grounding",
        "prompt": "What is the latest Indoone release?",
        "expected_topics": ["latest"],
    }
    score = score_case(
        case,
        "The latest Indoone release is version 2. Sources: https://example.com/release",
    )
    assert score["passed"] is True
    assert score["quality_passed"] is True
    assert score["quality_reason"] == ""


def test_score_case_enforces_supplied_grounding_evidence() -> None:
    case = {
        "id": "grounding",
        "category": "grounding",
        "prompt": "Use only the supplied evidence.",
        "expected_topics": ["SQLite"],
        "evidence": [
            {
                "title": "Conversation storage",
                "url": "",
                "snippet": "Indoone stores conversation messages in a local SQLite store.",
            }
        ],
    }
    supported = score_case(case, "The evidence says Indoone stores conversation messages in SQLite.")
    assert supported["passed"] is True
    assert supported["grounding_passed"] is True

    unsupported = score_case(case, "The evidence says Indoone stores conversation messages in PostgreSQL.")
    assert unsupported["passed"] is False
    assert unsupported["grounding_passed"] is True
    assert unsupported["forbidden_matches"] == []


def test_score_case_rejects_unsupported_grounding_fact() -> None:
    case = {
        "id": "grounding",
        "category": "grounding",
        "prompt": "Use only the supplied evidence.",
        "expected_topics": ["price"],
        "evidence": [
            {
                "title": "Pricing",
                "url": "",
                "snippet": "Indoone Pro costs 499 rupees per month.",
            }
        ],
    }
    score = score_case(case, "Indoone Pro costs 599 rupees per month.")
    assert score["passed"] is False
    assert score["grounding_passed"] is False
    assert score["grounding_reason"] == "unsupported_concrete_fact"


def test_build_case_prompt_preserves_multi_turn_context() -> None:
    case = {
        "prompt": "Now explain the topic concisely.",
        "turns": [
            {"role": "user", "content": "Please be concise."},
            {"role": "assistant", "content": "I will be concise."},
            {"role": "user", "content": "Now explain the topic concisely."},
        ],
    }
    prompt = build_case_prompt(case)
    assert prompt == (
        "User: Please be concise.\n"
        "Assistant: I will be concise.\n"
        "User: Now explain the topic concisely."
    )


def test_score_case_accepts_multi_turn_conversation_response() -> None:
    case = {
        "id": "conversation",
        "category": "conversation",
        "prompt": "Now explain the topic concisely.",
        "turns": [
            {"role": "user", "content": "Please answer concisely."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "Now explain the topic concisely."},
        ],
        "expected_topics": ["attention", "tokens", "concise"],
    }
    score = score_case(case, "A concise Transformer explanation uses attention over tokens.")
    assert score["passed"] is True


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
