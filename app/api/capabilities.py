from __future__ import annotations

import base64
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.ai.agent import MAX_AGENT_STEPS, execute_agent
from app.ai.research import build_research_provider
from app.capabilities.data_analysis import analyze_payload
from app.capabilities.document_extract import extract_document
from app.capabilities.image_generation import generate_image
from app.capabilities.integrations import get_integration, list_integrations
from app.capabilities.media import save_media
from app.capabilities.registry import list_capabilities
from app.capabilities.store import (
    create_project,
    create_task,
    delete_memory,
    delete_project,
    get_agent_run,
    get_project,
    list_agent_runs,
    list_memories,
    list_projects,
    list_tasks,
    search_memories,
    search_projects,
    update_project,
    upsert_memory,
)
from app.capabilities.vision import analyze_image
from app.capabilities.voice import synthesize_speech, transcribe_audio
from app.capabilities.voice_session import VoiceSessionState, handle_audio_message, handle_speak_message, normalize_request_id, ready_event

router = APIRouter(tags=["capabilities"])


class MemoryRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    key: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=4_000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="user", min_length=1, max_length=128)


class MemoryDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    memory_id: str = Field(min_length=1, max_length=128)


class ProjectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=200)
    instructions: str = Field(default="", max_length=8_000)
    context: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdateRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    instructions: str | None = Field(default=None, max_length=8_000)
    context: dict[str, Any] | None = None
    archived: bool | None = None


class ProjectDeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    project_id: str = Field(min_length=1, max_length=128)


class TaskRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=8_000)
    schedule: str = Field(min_length=1, max_length=500)
    enabled: bool = True


class AnalysisRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1, max_length=3_000_000)


class MediaRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class DocumentRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class VisionRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=12_000_000)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=5, ge=1, le=20)


class DeepResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1_000)
    queries: int = Field(default=3, ge=1, le=5)
    per_query_limit: int = Field(default=5, ge=1, le=10)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(default="", max_length=256)
    approved_tools: list[str] = Field(default_factory=list, max_length=8)


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4_000)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)


class VoiceTranscriptionRequest(BaseModel):
    audio_base64: str = Field(min_length=1, max_length=16_000_000)
    mime_type: str = Field(default="audio/wav", min_length=1, max_length=120)
    language: str = Field(default="", max_length=32)


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8_000)
    language: str = Field(default="", max_length=32)
    voice: str = Field(default="", max_length=128)
    format: str = Field(default="wav", min_length=1, max_length=16)


@router.get("/capabilities")
async def capabilities() -> dict[str, object]:
    return {"capabilities": list_capabilities()}


@router.get("/integrations")
async def integrations() -> dict[str, object]:
    return {"integrations": list_integrations()}


@router.get("/integrations/{integration_id}")
async def integration(integration_id: str) -> dict[str, object]:
    item = get_integration(integration_id)
    if item is None:
        raise HTTPException(status_code=404, detail="integration not found")
    return {"integration": item}


@router.get("/memory")
async def get_memories(
    user_id: str = Query(..., min_length=1, max_length=256),
    key_prefix: str = Query(default="", max_length=128),
    source: str = Query(default="", max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
    q: str = Query(default="", max_length=500),
) -> dict[str, object]:
    memories = search_memories(user_id, q, limit) if q.strip() else list_memories(user_id, key_prefix, source, limit)
    return {"memories": memories}


@router.post("/memory")
async def write_memory(request: MemoryRequest) -> dict[str, object]:
    return {"memory": upsert_memory(request.user_id, request.key, request.value, request.confidence, request.source)}


@router.delete("/memory")
async def remove_memory(request: MemoryDeleteRequest) -> dict[str, bool]:
    deleted = delete_memory(request.user_id, request.memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": True}


@router.post("/projects")
async def create_project_endpoint(request: ProjectRequest) -> dict[str, object]:
    return {"project": create_project(request.user_id, request.name, request.instructions, request.context)}


@router.get("/projects")
async def get_projects(user_id: str = Query(..., min_length=1, max_length=256), include_archived: bool = False, limit: int = Query(default=100, ge=1, le=500), q: str = Query(default="", max_length=500)) -> dict[str, object]:
    projects = search_projects(user_id, q, limit) if q.strip() else list_projects(user_id, include_archived, limit)
    return {"projects": projects}


@router.get("/projects/{project_id}")
async def get_project_endpoint(project_id: str, user_id: str = Query(..., min_length=1, max_length=256)) -> dict[str, object]:
    project = get_project(user_id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return {"project": project}


@router.patch("/projects/{project_id}")
async def update_project_endpoint(project_id: str, request: ProjectUpdateRequest) -> dict[str, object]:
    project = update_project(request.user_id, project_id, request.name, request.instructions, request.context, request.archived)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return {"project": project}


@router.delete("/projects/{project_id}")
async def delete_project_endpoint(project_id: str, request: ProjectDeleteRequest) -> dict[str, bool]:
    deleted = delete_project(request.user_id, project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="project not found")
    return {"deleted": True}


@router.post("/tasks")
async def create_task_endpoint(request: TaskRequest) -> dict[str, object]:
    return {"task": create_task(request.user_id, request.title, request.prompt, request.schedule, request.enabled)}


@router.get("/tasks")
async def get_tasks(user_id: str = Query(..., min_length=1, max_length=256)) -> dict[str, object]:
    return {"tasks": list_tasks(user_id)}


@router.get("/agent/runs")
async def agent_runs(user_id: str = Query(..., min_length=1, max_length=256), limit: int = Query(default=50, ge=1, le=100)) -> dict[str, object]:
    return {"runs": list_agent_runs(user_id, limit)}


@router.get("/agent/runs/{run_id}")
async def agent_run(run_id: str, user_id: str = Query(..., min_length=1, max_length=256)) -> dict[str, object]:
    run = get_agent_run(user_id, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="agent run not found")
    return {"run": run}


@router.post("/analysis")
async def analyze(request: AnalysisRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        return {"analysis": analyze_payload(request.filename, content)}
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/documents/extract")
async def extract_document_endpoint(request: DocumentRequest) -> dict[str, object]:
    try:
        content = base64.b64decode(request.content_base64, validate=True)
        result = extract_document(request.filename, request.mime_type, content)
        return {"document": {"filename": result.filename, "mime_type": result.mime_type, "page_count": result.page_count, "paragraphs": result.paragraphs, "characters": result.characters, "text": result.text, "truncated": result.truncated, "ocr_required": not bool(result.text)}}
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/vision")
async def vision(request: VisionRequest) -> dict[str, object]:
    try:
        return {"vision": analyze_image(request.filename, request.mime_type, request.content_base64)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/media")
async def upload_media(request: MediaRequest) -> dict[str, object]:
    try:
        return save_media(request.filename, request.mime_type, request.content_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/research")
async def research(request: ResearchRequest) -> dict[str, object]:
    provider = build_research_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="live research provider is not configured")
    try:
        results = await provider.search(request.query, request.limit)
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        raise HTTPException(status_code=502, detail=f"research provider failed: {exc}") from exc
    return {"query": request.query, "results": [{"title": item.title, "url": item.url, "snippet": item.snippet} for item in results]}


@router.post("/deep-research")
async def deep_research(request: DeepResearchRequest) -> dict[str, object]:
    provider = build_research_provider()
    if provider is None:
        raise HTTPException(status_code=503, detail="live research provider is not configured")
    queries = [request.query, f"{request.query} official sources", f"{request.query} recent developments"][: request.queries]
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            batch = await provider.search(query, request.per_query_limit)
        except (RuntimeError, ValueError, httpx.HTTPError) as exc:
            raise HTTPException(status_code=502, detail=f"research provider failed: {exc}") from exc
        for item in batch:
            if item.url in seen:
                continue
            seen.add(item.url)
            results.append({"query": query, "title": item.title, "url": item.url, "snippet": item.snippet})
    return {"query": request.query, "queries": queries, "sources": results}


@router.post("/agent")
async def agent(request: AgentRequest) -> dict[str, object]:
    execution = execute_agent(request.message, user_id=request.user_id, approved_tools=frozenset(request.approved_tools))
    if not execution.steps and not execution.blocked_steps:
        return {"intent": "general", "tool": None, "tool_payload": None, "result": None, "steps": [], "results": [], "memories": list(execution.memories), "blocked_steps": [], "retry_counts": list(execution.retry_counts), "run_id": execution.run_id, "max_steps": MAX_AGENT_STEPS}
    first = execution.steps[0] if execution.steps else execution.blocked_steps[0]
    first_result = execution.results[0] if execution.results else None
    return {"intent": "tool", "tool": first.tool, "tool_payload": first.payload, "result": None if first_result is None else {"name": first_result.name, "output": first_result.output, "safe": first_result.safe}, "steps": [{"index": step.index, "tool": step.tool, "payload": step.payload, "requires_approval": step.requires_approval} for step in execution.steps], "results": [{"name": result.name, "output": result.output, "safe": result.safe} for result in execution.results], "memories": list(execution.memories), "blocked_steps": [{"index": step.index, "tool": step.tool, "payload": step.payload, "requires_approval": step.requires_approval} for step in execution.blocked_steps], "retry_counts": list(execution.retry_counts), "run_id": execution.run_id, "max_steps": MAX_AGENT_STEPS}


@router.post("/image-generation")
async def image_generation(request: ImageGenerationRequest) -> dict[str, object]:
    try:
        result = await generate_image(request.prompt, request.width, request.height)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"result": {"provider": result.provider, "model": result.model, "mime_type": result.mime_type, "image_base64": result.image_base64, "metadata": result.metadata}}


@router.post("/voice/transcribe")
async def voice_transcribe(request: VoiceTranscriptionRequest) -> dict[str, object]:
    try:
        result = await transcribe_audio(request.audio_base64, mime_type=request.mime_type, language=request.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"result": {"text": result.text, "language": result.language, "provider": result.provider, "mime_type": result.mime_type, "metadata": result.metadata}}


@router.post("/voice/synthesize")
async def voice_synthesize(request: VoiceSynthesisRequest) -> dict[str, object]:
    try:
        result = await synthesize_speech(request.text, language=request.language, voice=request.voice, output_format=request.format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"result": {"audio_base64": result.audio_base64, "mime_type": result.mime_type, "provider": result.provider, "voice": result.voice, "metadata": result.metadata}}
