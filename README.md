# Indoone Backend

Backend foundation for the Indoone AI + automation platform.

## AI direction

Indoone's AI core is designed to run on Indoone-owned model code and local checkpoints. No hosted AI provider is required by the chat service.

The current local stack is:

`/api/chat` → AI service → local model runtime → Indoone Transformer checkpoint

## Project structure

- `app/api/` — HTTP API routes
- `app/ai/` — tokenizer, Transformer model, training, and local inference
- `data/` — starter training corpus
- `models/` — generated local checkpoints (ignored by Git)
- `tests/` — API and model tests

## Train the first model

```bash
python -m pip install -r requirements.txt
python -m app.ai.train --corpus data/indoone_corpus.txt --output models/indoone-small --steps 2000
```

Training writes a tokenizer, checkpoint, and metadata under `models/indoone-small/`.

## Run the backend

```bash
uvicorn app.main:app --reload
```

Health check: `GET /health`

Chat endpoint: `POST /api/chat` with `{ "message": "Hello Indoone" }`

Until a trained checkpoint exists, the API uses a deterministic local development fallback. Once the checkpoint is present, the same `/api/chat` endpoint automatically uses the Indoone local model runtime.

## Important

The starter corpus is only for validating the model-training and inference pipeline. A production-quality model requires a much larger, carefully licensed and curated dataset, stronger evaluation, safety testing, and substantially more training compute.
