from app.ai.research import ResearchResult, build_deep_research_queries, merge_research_results


def test_build_deep_research_queries_is_deterministic() -> None:
    assert build_deep_research_queries("  Indoone AI  ", 4) == [
        "Indoone AI",
        "Indoone AI official sources",
        "Indoone AI recent developments",
        "Indoone AI data statistics evidence",
    ]


def test_merge_research_results_deduplicates_and_keeps_provenance() -> None:
    merged = merge_research_results(
        {
            "q1": [ResearchResult("A", "https://example.com/a", "short")],
            "q2": [ResearchResult("A", "https://example.com/a", "a longer snippet")],
            "q3": [ResearchResult("B", "https://example.com/b", "other")],
        }
    )
    assert merged[0]["url"] == "https://example.com/a"
    assert merged[0]["queries"] == ["q1", "q2"]
    assert merged[0]["snippet"] == "a longer snippet"
    assert merged[1]["url"] == "https://example.com/b"
