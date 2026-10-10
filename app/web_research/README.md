# Indoone Web Research

This directory is the canonical home for live web research, source attribution, and evidence-grounding code and tests.

- `google_search.py` configures native Google Search grounding for supported Gemma 4 models, validates grounding metadata URLs, and formats source links. The model may decide when a search is useful based on the question and system instructions.
- `research.py` contains the optional public-source provider layer, query planning, relevance filtering, result merging, and evidence formatting. Its free provider uses public sources such as Wikipedia, Wikidata, Google News RSS, OpenAlex, and Crossref; the optional Tavily provider is used only when configured.
- `grounding.py` checks concrete facts and source URLs against supplied evidence and produces deterministic source attribution.
- `api.py` exposes the standalone `/api/research` and `/api/deep-research` endpoints.
- `tests/` contains research, grounding, API, source-propagation, and timeout-safety tests.

The general AI service and channel handlers should call this package through small interfaces. Keep provider, source, and evidence-specific logic here instead of duplicating it in chat or integration modules.

No search API secret is sent to clients. Gemini credentials remain server-side; public-source research remains available for standalone research routes and as an optional provider layer.
