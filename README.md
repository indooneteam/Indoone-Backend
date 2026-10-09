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


## Live Backend Control Center

The Control Center connects to these protected server endpoints:

- `GET /api/control-center/status` — effective global and per-channel settings.
- `PATCH /api/control-center/settings` — persist global intake/reply switches and per-channel intake/reply switches.
- `GET /api/control-center/metrics` — request and reply counters by channel (default window: 24 hours; `window_hours` supports 1–720).
- `GET /api/control-center/activity` — recent privacy-safe request/reply outcomes (route, status, HTTP code, time).

All Control Center endpoints require `Authorization: Bearer <INDOONE_CONTROL_CENTER_ADMIN_TOKEN>`. The dedicated token must be configured on the backend host and be at least 32 characters. It is separate from ordinary app-user tokens. Never commit it, send it in chat, or expose it through a `VITE_*` frontend variable.

Configure `INDOONE_ALLOWED_ORIGINS` to the exact dashboard origin, such as `https://indooneteam.github.io`. This variable is a comma-separated list and must not contain `*` in production. Configure the frontend with `VITE_INDOONE_API_BASE_URL` at build time or enter the HTTPS backend origin on the Control Center sign-in page.

### Pause behavior

- **Pause all app requests** leaves the server process and Control Center API running. Normal application `/api/` handlers are blocked with `503 INTAKE_PAUSED`; provider webhooks still validate their signature/secret and receive a safe acknowledgement without being processed or queued.
- Each app has its own intake switch. Turning off WhatsApp intake does not turn off Instagram, Telegram, or Android intake.
- **Pause AI replies** is independent from intake. Incoming events can still be accepted and counted, but an automated reply is skipped and recorded. WhatsApp, Instagram, Telegram, and Android reply switches are independent.
- Metrics begin recording after this feature is deployed; older requests are not reconstructed. Message contents, access tokens, customer phone numbers, and provider payloads are not stored in the Control Center activity table.

The repository change does not configure the live server automatically. Set the dedicated token and allowed origin in the server's private environment, then deploy/restart the backend and verify the protected status endpoint before using the dashboard.

## Important

The current starter corpus is only for validating the data, training, and inference pipeline. A production-quality model requires a much larger, carefully licensed and curated dataset, stronger evaluation, safety testing, and substantially more training compute.
