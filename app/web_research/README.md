# Indoone Web Research

This directory owns the live web research implementation and research-specific tests.

- `research.py` contains search providers, query variants, relevance checks, source merging, and evidence formatting.
- `tests/` contains provider, source-formatting, and timeout-safety tests.

AI-runtime and channel modules should call this package through a narrow interface; search-provider code should not be duplicated in those modules.

`app/ai/research.py` is temporarily retained as an import-only compatibility shim for the existing server-side integration. New code should import from `app.web_research`.
