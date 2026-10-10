# Indoone Web Research

This directory owns the live web research implementation and research-specific tests.

- `research.py` contains external search providers, query variants, relevance checks, source merging, and evidence formatting.
- `google_search.py` enables native Google Search grounding for supported Gemma 4 requests and extracts verified source URLs from grounding metadata.
- `tests/` contains provider, source-formatting, and timeout-safety tests.

AI-runtime and channel modules should call this package through a narrow interface; search-provider code should not be duplicated in those modules.

`app/ai/research.py` is temporarily retained as an import-only compatibility shim for the existing server-side integration. New code should import from `app.web_research`.
