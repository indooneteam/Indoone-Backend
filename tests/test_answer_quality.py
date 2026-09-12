import pytest

from app.ai.answer_quality import assess_answer, user_safe_failure


@pytest.mark.parametrize(
    "question, answer",
    [
        ("What is Indoone?", "Indoone is an AI platform."),
        ("Explain gravity.", "Gravity is the attraction between masses."),
    ],
)
def test_normal_answers_pass(question: str, answer: str) -> None:
    result = assess_answer(question, answer)
    assert result.passed is True


def test_internal_fallback_details_are_rejected() -> None:
    result = assess_answer(
        "What is Indoone?",
        "Indoone backend is reachable, but the trained local model is not available yet.",
    )
    assert result.passed is False
    assert result.reason == "internal_detail_leak"


def test_current_questions_need_sources() -> None:
    result = assess_answer(
        "What is the latest news?",
        "I don't have enough fresh information to confirm that.",
    )
    assert result.passed is False
    assert result.reason == "freshness_not_supported"


def test_current_questions_with_sources_pass() -> None:
    result = assess_answer(
        "What is the latest news?",
        "The latest report says the service launched a new update.\n\nSources:\n1. Example — https://example.com",
    )
    assert result.passed is True


def test_repeated_output_is_rejected() -> None:
    answer = "Same answer. Same answer. Same answer."
    result = assess_answer("Tell me something.", answer)
    assert result.passed is False
    assert result.reason == "repeated_output"


def test_safe_failure_message_is_user_facing() -> None:
    message = user_safe_failure()
    assert "reliable information" in message
    assert "fallback mode" not in message
