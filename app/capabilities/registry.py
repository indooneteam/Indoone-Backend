from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Capability:
    id: str
    name: str
    status: str
    backend_ready: bool
    app_integration: str
    notes: str


CAPABILITIES: tuple[Capability, ...] = (
    Capability("chat", "AI chat", "active", True, "ready", "Central chat endpoint and context pipeline."),
    Capability("chat_streaming", "Streaming chat", "active", True, "ready", "Server-sent events endpoint."),
    Capability("multilingual", "Indian multilingual support", "active", True, "ready", "Language detection and multilingual model/data path."),
    Capability("voice", "Real-time voice", "partial", True, "planned-ui", "Local/self-hosted STT and TTS adapters plus WebSocket voice session; model runtimes are configured separately."),
    Capability("memory", "Long-term memory", "partial", True, "planned-ui", "Durable user-scoped memory CRUD and candidate extraction."),
    Capability("projects", "Persistent projects/workspaces", "partial", True, "planned-ui", "Project metadata/context container backend."),
    Capability("tasks", "Scheduled tasks", "partial", True, "planned-ui", "Task definitions are persisted; an external worker is still required for time-based execution."),
    Capability("files", "Text/structured file context", "active", True, "existing", "Current text/JSON/CSV file-aware chat path."),
    Capability("documents", "PDF/DOCX ingestion contract", "partial", True, "planned-ui", "Binary media intake contract and text extraction."),
    Capability("vision", "Image OCR and vision intake", "partial", True, "planned-ui", "Local OCR and image metadata are active; a dedicated semantic vision model is still required for full image understanding."),
    Capability("image_generation", "Image generation", "partial", True, "planned-ui", "Provider/model adapter contract; generation requires a configured local model runtime."),
    Capability("data_analysis", "CSV/JSON data analysis", "active", True, "planned-ui", "Safe standard-library summary/statistics API."),
    Capability("research", "Live research", "partial", True, "planned-ui", "Controlled HTTP research provider; enabled only when configured."),
    Capability("deep_research", "Multi-step research", "partial", True, "planned-ui", "Multi-query source collection built on the research provider."),
    Capability("tools", "Safe tools", "active", True, "existing", "Calculator/orchestrator tool layer."),
    Capability("agent", "Agent planning/execution", "active", True, "planned-ui", "Bounded deterministic multi-step execution with explicit step/results contract; broader autonomous tool selection remains gated."),
    Capability("canvas", "Canvas/work documents", "partial", True, "planned-ui", "Project document contract; rich editor remains app-side."),
    Capability("connectors", "External app connectors", "planned", True, "planned-ui", "Connector registry contract without live third-party credentials."),
    Capability("evaluation", "AI quality/evaluation", "active", True, "internal", "Behavioral and answer-quality checks are already part of the pipeline."),
    Capability("training", "Custom model training", "active", True, "internal", "Validated training pipeline with dataset/readiness gates."),
)


def list_capabilities() -> list[dict[str, object]]:
    return [asdict(item) for item in CAPABILITIES]
