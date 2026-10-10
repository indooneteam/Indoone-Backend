# Indoone Web Research

This directory owns the live web research implementation and research-specific tests.

- `research.py` contains external search providers, query variants, relevance checks, source merging, and evidence formatting.
- `google_search.py` detects live-search intent and formats source links. Search uses only keyless public providers: Wikipedia, Wikidata, Google News RSS, OpenAlex, and Crossref; no paid search API is required.
- `tests/` contains provider, source-formatting, and timeout-safety tests.

AI-runtime and channel modules should call this package through a narrow interface; search-provider code should not be duplicated in those modules.

The implementation has one canonical location: `app/web_research/`. Repository callers should import research functionality from this package; do not reintroduce the implementation under `app/ai/` or channel-specific modules.
