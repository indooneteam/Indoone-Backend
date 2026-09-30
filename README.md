# Indoone Backend

Backend foundation for the Indoone AI + automation platform.

## AI direction

Indoone's AI core is designed to run on Indoone-owned model code and local checkpoints. No hosted AI provider is required by the chat service.

The current local stack is:

`/api/chat` → AI service → local model runtime → Indoone Transformer checkpoint

When configured, fresh-information requests can also use the optional research layer:

`/api/chat` → AI service → research provider → source results → local AI model

Research is separate from model inference. It returns source title, URL, and optional snippet so provenance is retained in the model context.

## Live research

Fresh-information requests use live research before local model synthesis. When `INDOONE_RESEARCH_URL` is configured, Indoone uses that custom search endpoint. When `INDOONE_TAVILY_API_KEY` is configured, Tavily becomes the primary general web-search provider and returns ranked web results for grounded synthesis. If neither is configured, Indoone falls back to the built-in multi-source research providers.

```text
INDOONE_TAVILY_API_KEY=
INDOONE_RESEARCH_URL=https://your-search-service.example/search
INDOONE_RESEARCH_TOKEN=
INDOONE_RESEARCH_TIMEOUT=10
```

The Tavily integration uses basic search by default to keep API-credit usage predictable. Tavily's current free plan provides 1,000 API credits per month and does not require a credit card. Basic search costs 1 credit per search; advanced search costs 2 credits. citeturn992854search0turn992854search6

The backend invokes research for messages that indicate freshness or research needs, such as "latest", "current", "today", "news", or "research". Returned source titles, URLs, and snippets are passed into local model context for grounded synthesis.

## Dataset pipeline

Place source documents in `data/raw/`. The preparation script normalizes whitespace, splits documents, removes very short entries, removes exact duplicate content using a normalized SHA-256 fingerprint, and creates deterministic train/validation/test files under `data/processed/`.

```bash
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
```

The processed dataset is a generated training artifact and should not be committed as a substitute for the original licensed source data.

## Project structure

- `app/api/` — HTTP API routes
- `app/ai/` — local AI runtime and domain modules; `app/ai/training/` contains model training and evaluation
- `data/raw/` — source training documents
- `data/knowledge/` — approved local knowledge sources
- `data/processed/` — generated train/validation/test splits
- `models/` — generated local checkpoints (ignored by Git)
- `scripts/` — dataset, training, and runtime utilities
- `tests/` — API, dataset, AI, and integration tests

## Train the first model

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
python -m app.ai.training.train --corpus data/processed/train.txt --output models/indoone-small --steps 2000
```

Training writes a tokenizer, checkpoint, and metadata under `models/indoone-small/`.

## Run the backend

Development:

```bash
uvicorn app.main:app --reload
```

Production/deployment:

```bash
python -m scripts.run_server
```

The production entrypoint honors a deployment-provided `PORT`, defaults to one worker, disables reload in production, bounds concurrency and keep-alive/graceful-shutdown settings, suppresses the Uvicorn `Server` header, and keeps Uvicorn-level forwarded-header trust disabled. The application only enables forwarded-header handling through its explicit trusted-proxy configuration.

Health check: `GET /health`

Readiness check: `GET /ready`

Detailed health: `GET /health/details`

Chat endpoint: `POST /api/chat` with `{ "message": "Hello Indoone" }`

Until a trained checkpoint exists, the API returns a clear local-model-unavailable error. Once the checkpoint is present, the same `/api/chat` endpoint uses the Indoone local model runtime.

## Important

The current starter corpus is only for validating the data, training, and inference pipeline. A production-quality model requires a much larger, carefully licensed and curated dataset, stronger evaluation, safety testing, and substantially more training compute.
