# Indoone Backend

Backend foundation for the Indoone AI + automation platform.

## AI direction

Indoone chat keeps local-model inference as the default:

`/api/chat` → configured provider → local Indoone model (default) or Google Gemini (when explicitly enabled)

Set `INDOONE_MODEL_BACKEND=gemini` to route chat through Google AI Studio/Gemini. In this mode, supported Gemma 4 models can decide whether a question needs live web research. When needed, the backend runs the bounded `search_web` tool against keyless public sources (Wikipedia, Wikidata, Google News RSS, OpenAlex, and Crossref), passes the evidence back to the model, and includes validated source links in the chat response. No separate paid search-provider API key is required for this free provider set. Stable questions can be answered without a search.

The Gemini key stays server-side in `GEMINI_API_KEY`; the chat provider is not enabled merely because a key exists. Explicitly configure `INDOONE_MODEL_BACKEND=gemini` and a supported `GEMINI_MODEL` to activate it. If the variable is absent or set to `local`, the existing local-model answer path remains unchanged.

The trained model artifacts are loaded only from the private `Indoone-Model` GitHub Release configured by:

```text
GITHUB_MODEL_REPOSITORY=indooneteam/Indoone-Model
GITHUB_MODEL_RELEASE_TAG=indoone-model-v1
GITHUB_MODEL_TOKEN=
```

Model generation has no application-level timeout. Render Web Services allow HTTP responses to run for up to 100 minutes, so a one-hour model generation target remains within the platform request limit. citeturn301152search0turn301152search6

## Dataset pipeline

Place source documents in `data/raw/`. The preparation script normalizes whitespace, splits documents, removes very short entries, removes exact duplicate content using a normalized SHA-256 fingerprint, and creates deterministic train/validation/test files under `data/processed/`.

```bash
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
```

The processed dataset is a generated training artifact and should not be committed as a substitute for the original licensed source data.

## Project structure

- `app/api/` — HTTP API routes
- `app/ai/` — local AI runtime and domain modules; `app/ai/training/` contains model training and evaluation
- `app/web_research/` — free public search providers, model-selected search tool, evidence validation, source formatting, and research-specific tests
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
