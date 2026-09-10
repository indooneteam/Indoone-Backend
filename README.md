# Indoone Backend

Backend foundation for the Indoone AI + automation platform.

## AI direction

Indoone's AI core is designed to run on Indoone-owned model code and local checkpoints. No hosted AI provider is required by the chat service.

The current local stack is:

`/api/chat` → AI service → local model runtime → Indoone Transformer checkpoint

## Dataset pipeline

Place source documents in `data/raw/`. The preparation script normalizes whitespace, splits documents, removes very short entries, removes exact duplicate content using a normalized SHA-256 fingerprint, and creates deterministic train/validation/test files under `data/processed/`.

```bash
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
```

The processed dataset is a generated training artifact and should not be committed as a substitute for the original licensed source data.

## Project structure

- `app/api/` — HTTP API routes
- `app/ai/` — tokenizer, Transformer model, training, and local inference
- `data/raw/` — source training documents
- `data/processed/` — generated train/validation/test splits
- `models/` — generated local checkpoints (ignored by Git)
- `scripts/` — dataset and training utilities
- `tests/` — API, dataset, and model tests

## Train the first model

```bash
python -m pip install -r requirements.txt
python -m scripts.prepare_dataset --source data/raw/indoone_corpus.txt --output-dir data/processed
python -m app.ai.train --corpus data/processed/train.txt --output models/indoone-small --steps 2000
```

Training writes a tokenizer, checkpoint, and metadata under `models/indoone-small/`.

## Run the backend

```bash
uvicorn app.main:app --reload
```

Health check: `GET /health`

Chat endpoint: `POST /api/chat` with `{ "message": "Hello Indoone" }`

Until a trained checkpoint exists, the API returns a clear local-model-unavailable error. Once the checkpoint is present, the same `/api/chat` endpoint uses the Indoone local model runtime.

## Important

The current starter corpus is only for validating the data, training, and inference pipeline. A production-quality model requires a much larger, carefully licensed and curated dataset, stronger evaluation, safety testing, and substantially more training compute.
