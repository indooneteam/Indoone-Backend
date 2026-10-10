from app.web_research.google_search import (
    build_google_search_tools,
    extract_grounding_sources,
    format_sources_footer,
    split_sources_footer,
)
from app.web_research.research import ResearchResult


def test_google_search_grounding_is_enabled_for_both_gemma4_models() -> None:
    assert build_google_search_tools("gemma-4-26b-a4b-it") == [{"googleSearch": {}}]
    assert build_google_search_tools("gemma-4-31b-it") == [{"googleSearch": {}}]


def test_google_search_grounding_is_not_added_to_unconfigured_models() -> None:
    assert build_google_search_tools("gemini-2.5-flash") == []
    assert build_google_search_tools(" custom-model ") == []


def test_extract_grounding_sources_validates_urls_and_deduplicates() -> None:
    response = {
        "candidates": [{
            "groundingMetadata": {
                "groundingChunks": [
                    {"web": {"title": "Latest AI news", "uri": "https://example.com/news"}},
                    {"web": {"title": "Latest AI news duplicate", "uri": "https://example.com/news/"}},
                    {"web": {"title": "Bad URL", "uri": "javascript:alert(1)"}},
                    {"web": {"title": "", "uri": "https://example.com/no-title"}},
                ]
            }
        }]
    }
    sources = extract_grounding_sources(response)
    assert sources == [ResearchResult("Latest AI news", "https://example.com/news", "")]


def test_extract_grounding_sources_handles_missing_metadata() -> None:
    assert extract_grounding_sources({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}) == []
    assert extract_grounding_sources({}) == []


def test_format_sources_footer_appends_only_valid_sources() -> None:
    answer = format_sources_footer(
        "Here is the latest answer.",
        [
            ResearchResult("Official update", "https://example.com/update", ""),
            ResearchResult("Invalid", "javascript:alert(1)", ""),
        ],
    )
    assert answer.startswith("Here is the latest answer.\n\nSources:\n")
    assert "- [Official update](https://example.com/update)" in answer
    assert "javascript:" not in answer


def test_format_sources_footer_leaves_answer_unchanged_without_sources() -> None:
    assert format_sources_footer("Answer.  ", []) == "Answer."


def test_split_sources_footer_returns_structured_sources() -> None:
    answer, sources = split_sources_footer(
        "Current answer.\n\nSources:\n- [Official update](https://example.com/update)\n- [Second source](https://example.org/news)"
    )
    assert answer == "Current answer."
    assert sources == [
        ResearchResult("Official update", "https://example.com/update", ""),
        ResearchResult("Second source", "https://example.org/news", ""),
    ]


def test_split_sources_footer_leaves_regular_or_invalid_text_unchanged() -> None:
    text = "Answer.\n\nSources:\n- [Unsafe](javascript:alert(1))"
    assert split_sources_footer("Just an answer.") == ("Just an answer.", [])
    assert split_sources_footer(text) == (text, [])
