from app.ai.grounding import (
    GroundedEvidence,
    append_sources,
    assess_grounding,
    build_grounded_prompt_instruction,
)


def test_grounded_prompt_instruction_requires_evidence_discipline() -> None:
    instruction = build_grounded_prompt_instruction()
    assert "Do not invent facts" in instruction
    assert "supplied evidence" in instruction


def test_append_sources_is_deterministic_and_deduplicates_urls() -> None:
    evidence = [
        GroundedEvidence("Example", "https://example.com", "one"),
        GroundedEvidence("Example duplicate", "https://example.com", "two"),
        GroundedEvidence("Second", "https://second.example", "three"),
    ]

    result = append_sources("Answer", evidence)

    assert result.count("https://example.com") == 1
    assert "1. Example — https://example.com" in result
    assert "2. Second — https://second.example" in result


def test_append_sources_without_evidence_returns_clean_answer() -> None:
    assert append_sources("  Answer  ", []) == "Answer"


def test_append_sources_rejects_internal_model_details() -> None:
    result = append_sources("The trained local model is not available yet.", [])
    assert "reliable information" in result
    assert "trained local model" not in result


def test_append_sources_rejects_repeated_output() -> None:
    result = append_sources("Same answer. Same answer. Same answer.", [])
    assert "reliable information" in result


def test_grounding_accepts_answer_supported_by_evidence() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    result = assess_grounding("Indoone Pro costs 499 rupees per month.", evidence)
    assert result.passed is True


def test_grounding_rejects_unsupported_concrete_fact() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    result = assess_grounding("Indoone Pro costs 599 rupees per month.", evidence)
    assert result.passed is False
    assert result.reason == "unsupported_concrete_fact"


def test_grounding_allows_nonnumeric_explanatory_prose() -> None:
    evidence = [GroundedEvidence("Pricing", "https://example.com/pricing", "Pricing information.")]
    result = assess_grounding("Here is a concise explanation.", evidence)
    assert result.passed is True


def test_grounding_rejects_unknown_source_url() -> None:
    evidence = [GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")]
    result = assess_grounding("Indoone Pro costs 499 rupees per month. https://attacker.example", evidence)
    assert result.passed is False
    assert result.reason == "unsupported_source_url"
