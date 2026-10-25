from __future__ import annotations

import base64
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.plugins.github.routes import router as github_router
from app.capabilities.integrations import (
    probe_integration,
)
from app.capabilities.store import create_oauth_state
from secrets import token_urlsafe
from app.api.security.errors import provider_failure

router = APIRouter(prefix="/integrations", tags=["integrations"])
router.include_router(github_router)


class IntegrationProbeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


@router.post("/{integration_id}/probe")
async def integration_probe(integration_id: str, request: IntegrationProbeRequest) -> dict[str, object]:
    try:
        return await probe_integration(request.user_id, integration_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise provider_failure("integration provider", exc) from exc
    except RuntimeError as exc:
        raise provider_failure("integration", exc, status_code=503) from exc


