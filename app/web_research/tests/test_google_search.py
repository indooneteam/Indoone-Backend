from app.web_research.google_search import (
    format_sources_footer,
    should_use_live_research,
    split_sources_footer,
)
from app.web_research.research import ResearchResult


def test_live_research_intent_is_selective_and_multilingual() -> None:
    assert should_use_live_research("What is the latest Indoone update?")
    assert should_use_live_research("ಇವತ್ತಿನ ಸುದ್ದಿ ಏನು?")
    assert should_use_live_research("ivattina suddi heli")
    assert not should_use_live_research("Explain photosynthesis simply")


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
    answer_text = "\n".join([
        "Current answer.",
        "",
        "Sources:",
        "- [Official update](https://example.com/update)",
        "- [Second source](https://example.org/news)",
    ])
    answer, sources = split_sources_footer(answer_text)
    assert answer == "Current answer."
    assert sources == [
        ResearchResult("Official update", "https://example.com/update", ""),
        ResearchResult("Second source", "https://example.org/news", ""),
    ]


def test_split_sources_footer_leaves_regular_or_invalid_text_unchanged() -> None:
    text = "\n".join(["Answer.", "", "Sources:", "- [Unsafe](javascript:alert(1))"])
    assert split_sources_footer("Just an answer.") == ("Just an answer.", [])
    assert split_sources_footer(text) == (text, [])
