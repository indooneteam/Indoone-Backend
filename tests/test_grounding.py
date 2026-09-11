from app.ai.grounding import GroundedEvidence, append_sources, build_grounded_prompt_instruction


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
