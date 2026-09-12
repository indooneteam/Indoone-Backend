from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.integrations import probe_integration

router = APIRouter(prefix="/integrations", tags=["integrations"])


class IntegrationProbeRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)


@router.post("/{integration_id}/probe")
async def integration_probe(integration_id: str, request: IntegrationProbeRequest) -> dict[str, object]:
    try:
        return await probe_integration(request.user_id, integration_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"integration provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
