import pytest

from app.web_research.grounding import (
    GroundedEvidence,
    append_sources,
    extract_sources,
    assess_grounding,
    build_grounded_prompt_instruction,
)


def test_grounded_prompt_instruction_requires_evidence_discipline() -> None:
    instruction = build_grounded_prompt_instruction()
    assert "Do not invent facts" in instruction
    assert "supplied evidence" in instruction
    assert "precise dates, quantities, percentages, and statistics" in instruction


def test_grounded_prompt_requires_one_synthesized_answer() -> None:
    instruction = build_grounded_prompt_instruction()
    assert "one coherent answer" in instruction
    assert "separate source-by-source answers" in instruction


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
    with pytest.raises(RuntimeError, match="quality checks"):
        append_sources("The trained local model is not available yet.", [])


def test_append_sources_rejects_repeated_output() -> None:
    with pytest.raises(RuntimeError, match="quality checks"):
        append_sources("Same answer. Same answer. Same answer.", [])


def test_grounding_accepts_answer_supported_by_evidence() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    result = assess_grounding("Indoone Pro costs 499 rupees per month.", evidence)
    assert result.passed is True


def test_grounding_flags_unsupported_concrete_fact_for_evaluation() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    result = assess_grounding("Indoone Pro costs 599 rupees per month.", evidence)
    assert result.passed is False
    assert result.reason == "unsupported_concrete_fact"


def test_append_sources_does_not_turn_numeric_mismatch_into_chat_failure() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    # The checker flags this mismatch for evaluation, but production must not return
    # HTTP 503 solely because compact search snippets don't prove a numerical claim.
    result = append_sources("Indoone Pro costs 599 rupees per month.", evidence)
    assert "costs 599 rupees" in result
    assert "https://example.com/pricing" in result


def test_grounding_rejects_unknown_url_even_when_a_numeric_mismatch_exists() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    result = assess_grounding(
        "Indoone Pro costs 599 rupees per month. https://attacker.example",
        evidence,
    )
    assert result.passed is False
    assert result.reason == "unsupported_source_url"


def test_append_sources_rejects_unknown_url_even_when_a_numeric_mismatch_exists() -> None:
    evidence = [
        GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")
    ]
    with pytest.raises(RuntimeError, match="unsupported_source_url"):
        append_sources(
            "Indoone Pro costs 599 rupees per month. https://attacker.example",
            evidence,
        )


def test_append_sources_keeps_research_answer_when_compact_snippets_omit_year() -> None:
    evidence = [
        GroundedEvidence("Research report", "https://example.com/report", "The report summarizes the project.")
    ]
    result = append_sources("The project update was published in 2025.", evidence)
    assert "published in 2025" in result
    assert "Sources:" in result
    assert "https://example.com/report" in result


def test_grounding_allows_nonnumeric_explanatory_prose() -> None:
    evidence = [GroundedEvidence("Pricing", "https://example.com/pricing", "Pricing information.")]
    result = assess_grounding("Here is a concise explanation.", evidence)
    assert result.passed is True


def test_grounding_rejects_unknown_source_url() -> None:
    evidence = [GroundedEvidence("Pricing", "https://example.com/pricing", "Indoone Pro costs 499 rupees per month.")]
    result = assess_grounding("Indoone Pro costs 499 rupees per month. https://attacker.example", evidence)
    assert result.passed is False
    assert result.reason == "unsupported_source_url"


def test_extract_sources_parses_deterministic_source_list() -> None:
    reply = (
        "One coherent answer.\n\nSources:\n"
        "1. First source — https://example.com/one\n"
        "2. Second source — https://example.org/two"
    )

    assert extract_sources(reply) == [
        GroundedEvidence("First source", "https://example.com/one"),
        GroundedEvidence("Second source", "https://example.org/two"),
    ]


def test_split_grounded_sources_returns_numbered_links_and_preserves_unparsed_text() -> None:
    from app.web_research.grounding import split_grounded_sources

    text = "\n".join([
        "Current answer.",
        "",
        "Sources:",
        "1. Official update — https://example.com/update",
        "2. Public report — https://example.org/report",
    ])
    answer, sources = split_grounded_sources(text)
    assert answer == "Current answer."
    assert [(source.title, source.url) for source in sources] == [
        ("Official update", "https://example.com/update"),
        ("Public report", "https://example.org/report"),
    ]
    assert split_grounded_sources("Just a simple answer.") == ("Just a simple answer.", [])
