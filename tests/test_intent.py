from app.ai.intent import classify_intent


def test_current_questions_use_fresh_research() -> None:
    intent = classify_intent("What is the latest India news today?")
    assert intent.name == "research"
    assert intent.needs_research is True
    assert intent.needs_cross_check is True


def test_dynamic_price_uses_cross_check() -> None:
    intent = classify_intent("What is the current price of this phone?")
    assert intent.needs_research is True
    assert intent.needs_cross_check is True


def test_stable_general_question_does_not_force_web() -> None:
    intent = classify_intent("Explain photosynthesis")
    assert intent.needs_research is False
    assert intent.needs_cross_check is False
