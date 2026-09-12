from __future__ import annotations

import base64
from typing import Any
import httpx
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from secrets import token_urlsafe
from app.ai.agent import MAX_AGENT_STEPS, execute_agent
from app.ai.research import build_research_provider
from app.capabilities.data_analysis import analyze_payload
from app.capabilities.document_extract import extract_document
from app.capabilities.image_generation import generate_image
from app.capabilities.integrations import build_oauth_authorization, exchange_oauth_code, get_integration, list_integrations, probe_integration
from app.capabilities.media import save_media
from app.capabilities.registry import list_capabilities
from app.capabilities.store import create_oauth_state, create_project, create_task, delete_memory, delete_project, get_agent_run, get_project, list_agent_runs, list_memories, list_projects, list_tasks, search_memories, search_projects, update_project, upsert_memory
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
class DocumentRequest(MediaRequest): pass
class VisionRequest(MediaRequest): pass
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
class IntegrationConnectRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    redirect_uri: str = Field(min_length=1, max_length=2_000)
class IntegrationCallbackRequest(BaseModel):
    state: str = Field(min_length=16, max_length=512)
    code: str = Field(min_length=1, max_length=8_000)
class IntegrationProbeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
@router.get("/capabilities")
async def capabilities() -> dict[str, object]: return {"capabilities": list_capabilities()}
@router.get("/integrations")
async def integrations() -> dict[str, object]: return {"integrations": list_integrations()}
@router.get("/integrations/{integration_id}")
async def integration(integration_id: str) -> dict[str, object]:
    item = get_integration(integration_id)
    if item is None: raise HTTPException(status_code=404, detail="integration not found")
    return {"integration": item}
@router.post("/integrations/{integration_id}/connect")
async def connect_integration(integration_id: str, request: IntegrationConnectRequest) -> dict[str, object]:
    state = token_urlsafe(32)
    try:
        create_oauth_state(state, request.user_id, integration_id.strip().lower(), request.redirect_uri.strip())
        result = build_oauth_authorization(integration_id, state, request.redirect_uri)
        return {**result, "state": state}
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc
@router.post("/integrations/{integration_id}/callback")
async def integration_callback(integration_id: str, request: IntegrationCallbackRequest) -> dict[str, object]:
    try: return await exchange_oauth_code(integration_id, request.state, request.code)
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, httpx.HTTPError) as exc: raise HTTPException(status_code=502, detail=f"oauth exchange failed: {exc}") from exc
@router.post("/integrations/{integration_id}/probe")
async def integration_probe(integration_id: str, request: IntegrationProbeRequest) -> dict[str, object]:
    try: return await probe_integration(request.user_id, integration_id)
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc: raise HTTPException(status_code=502, detail=f"integration provider failed: {exc}") from exc
    except RuntimeError as exc: raise HTTPException(status_code=503, detail=str(exc)) from exc
@router.get("/memory")
async def get_memories(user_id: str = Query(..., min_length=1, max_length=256), key_prefix: str = Query(default="", max_length=128), source: str = Query(default="", max_length=128), limit: int = Query(default=100, ge=1, le=500), q: str = Query(default="", max_length=500)) -> dict[str, object]:
    return {"memories": search_memories(user_id, q, limit) if q.strip() else list_memories(user_id, key_prefix, source, limit)}
@router.post("/memory")
async def write_memory(request: MemoryRequest) -> dict[str, object]: return {"memory": upsert_memory(request.user_id, request.key, request.value, request.confidence, request.source)}
@router.delete("/memory")
async def remove_memory(request: MemoryDeleteRequest) -> dict[str, bool]:
    if not delete_memory(request.user_id, request.memory_id): raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": True}
