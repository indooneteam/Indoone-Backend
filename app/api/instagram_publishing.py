from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.instagram_publishing import create_carousel_container, publish_ready_container

router = APIRouter(prefix="/integrations/instagram/publishing", tags=["instagram-publishing"])


class CarouselRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    image_urls: list[str] = Field(min_length=2, max_length=10)
    caption: str = Field(default="", max_length=2200)
    approved: bool = False


class PublishRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    creation_id: str = Field(min_length=1, max_length=256)
    approved: bool = False


@router.post("/carousel")
async def carousel(request: CarouselRequest) -> dict[str, object]:
    try:
        return await create_carousel_container(
            request.user_id,
            request.image_urls,
            request.caption,
            request.approved,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram publishing provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/publish-ready")
async def publish_ready(request: PublishRequest) -> dict[str, object]:
    try:
        return await publish_ready_container(request.user_id, request.creation_id, request.approved)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"instagram publishing provider failed: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
