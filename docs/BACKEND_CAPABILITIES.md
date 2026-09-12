# Indoone Backend Capability Foundation

This backend exposes stable contracts for the major assistant capabilities so the Android app can be upgraded independently later.

## Endpoints

- `GET /api/capabilities` — capability registry and readiness state.
- `GET/POST/DELETE /api/memory` — durable user-scoped memory records.
- `GET/POST /api/projects` — persistent project/workspace metadata and context.
- `GET/POST /api/tasks` — persistent task definitions and schedules.
- `POST /api/analysis` — safe CSV/JSON/JSONL summaries and numeric statistics.
- `POST /api/media` — image/PDF/DOCX binary intake metadata and storage.
- `POST /api/research` — controlled live research provider.
- `POST /api/deep-research` — multi-query source collection with provenance.
- `POST /api/agent` — orchestrator/tool planning contract.
- `POST /api/image-generation` — local image-model adapter contract.

## Important boundaries

The foundation does not pretend that a backend API contract is the same thing as a trained multimodal model or a scheduler worker. Vision inference, image generation, document extraction, live research, and task execution each require their respective runtime/model/provider to be configured.

User authentication must be connected to `user_id` before exposing memory, project, and task mutations to production clients. The current endpoints deliberately make ownership explicit so Android integration can be added without changing the storage model.
