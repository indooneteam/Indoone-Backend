# Indoone Web Research

This directory is the canonical home for live web research, source attribution, and evidence-grounding code and tests.

- `tools.py` exposes a Gemma 4 `search_web` function and executes bounded searches using keyless public providers. The model decides whether to call the tool.
- `research.py` contains public-source providers, query planning, relevance filtering, result merging, and evidence formatting. The free provider uses Wikipedia, Wikidata, Google News RSS, OpenAlex, and Crossref.
- `grounding.py` checks concrete facts and URLs against supplied evidence and provides deterministic source attribution.
- `api.py` exposes `/api/research` and `/api/deep-research`.
- `tests/` contains provider, tool-call, grounding, API, source-propagation, and timeout-safety tests.

No paid search-provider API key or Google Search grounding tool is used by chat. Gemini API usage itself may still be subject to its own quota/pricing. The model can answer stable questions directly; when it requests `search_web`, the backend searches only public sources, returns the evidence to Gemma, and adds verifiable source links to the final answer.

Keep provider, source, and evidence-specific logic here instead of duplicating it in chat or channel modules.
