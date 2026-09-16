from app.ai.response_style import assess_response_style


def test_concise_style_accepts_short_response() -> None:
    result = assess_response_style(
        "Explain attention concisely.",
        "Attention lets each token weigh other relevant tokens.",
    )
    assert result.passed is True
    assert "concise" in result.checks


def test_concise_style_rejects_overlong_response() -> None:
    result = assess_response_style(
        "Explain attention concisely.",
        " ".join(["Attention connects related tokens."] * 30),
    )
    assert result.passed is False
    assert result.reason == "style_not_concise"


def test_exactly_three_steps_requires_three_items() -> None:
    result = assess_response_style(
        "Give me exactly three simple steps.",
        "1. Plan\n2. Execute\n3. Review",
        profile="exactly_three_steps",
    )
    assert result.passed is True
    assert "exactly_three_items" in result.checks


def test_exactly_three_steps_rejects_four_items() -> None:
    result = assess_response_style(
        "Give me exactly three simple steps.",
        "1. Plan\n2. Execute\n3. Review\n4. Repeat",
        profile="exactly_three_steps",
    )
    assert result.passed is False
    assert result.reason == "style_exactly_three_items"


def test_bullet_constraint_rejects_plain_prose() -> None:
    result = assess_response_style(
        "Explain this as bullet points.",
        "First do this. Then do that.",
    )
    assert result.passed is False
    assert result.reason == "style_bullets_missing"


def test_no_markdown_constraint_rejects_markdown() -> None:
    result = assess_response_style(
        "Answer with no markdown.",
        "**Answer:** plain text",
    )
    assert result.passed is False
    assert result.reason == "style_markdown_not_allowed"
