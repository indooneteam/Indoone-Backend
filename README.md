# Indoone Backend

Backend foundation for the Indoone AI + automation platform.

## AI direction

Indoone's chat service uses one direct answer path:

`/api/chat` → local Indoone model provider → Indoone Transformer V1 checkpoint

For a normal chat request, the backend does not route the question through web research, retrieval, a canned answer, a fallback answer, or an external hosted LLM. The model receives the actual user request directly. Existing conversation history and an explicitly attached document are included only when the user supplies or continues that context.

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
- `app/web_research/` — isolated live web research providers, source handling, evidence formatting, and research-specific tests
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
