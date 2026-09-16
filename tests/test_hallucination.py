from app.ai.hallucination import assess_hallucination


def test_affirmative_external_action_claim_is_rejected_without_verification() -> None:
    result = assess_hallucination("I checked your private company database.")
    assert result.passed is False
    assert result.reason == "unsupported_external_action_claim"


def test_negated_external_action_claim_is_allowed() -> None:
    result = assess_hallucination("I cannot access or check your private company database.")
    assert result.passed is True


def test_verified_external_action_can_be_allowlisted() -> None:
    result = assess_hallucination(
        "I checked the connected calendar.",
        ["I checked"],
    )
    assert result.passed is True


def test_unverified_action_remains_rejected_with_other_allowlisted_action() -> None:
    result = assess_hallucination(
        "I checked the connected calendar and searched your private database.",
        ["I checked"],
    )
    assert result.passed is False
    assert result.reason == "unsupported_external_action_claim"
