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


def test_lightweight_understanding_handles_romanized_kannada() -> None:
    from app.ai.question_understanding import understand_question

    understood = understand_question("gravity andre enu?")
    assert understood.language == "Kannada"
    assert understood.intent == "general"
    assert understood.question_type == "general"
    assert understood.research_query == "gravity"


def test_kannada_definition_is_not_misclassified_as_research() -> None:
    from app.ai.question_understanding import understand_question

    understood = understand_question("ಗ್ರಾವಿಟಿ ಎಂದರೇನು")

    assert understood.language == "Kannada"
    assert understood.intent == "general"
    assert understood.needs_research is False


def test_lightweight_understanding_extracts_english_topic_from_question() -> None:
    from app.ai.question_understanding import understand_question

    understood = understand_question("What is gravity?")
    assert understood.language == "English"
    assert understood.question_type == "definition"
    assert understood.research_query == "gravity"


def test_lightweight_understanding_routes_current_question_without_model() -> None:
    from app.ai.question_understanding import understand_question

    understood = understand_question("what is the latest India news today?")
    assert understood.intent == "research"
    assert understood.needs_research is True
    assert understood.needs_cross_check is True


def test_lightweight_understanding_handles_typo_without_changing_unknown_topic() -> None:
    from app.ai.question_understanding import understand_question

    understood = understand_question("wat is gravty?")
    assert understood.intent == "general"
    assert understood.research_query == "wat gravty"
