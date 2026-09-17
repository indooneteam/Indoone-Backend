from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.capabilities.contacts import Contact, parse_contacts, resolve_contact, search_contacts

router = APIRouter(tags=["contacts"])


class ContactSearchRequest(BaseModel):
    contacts: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    query: str = Field(default="", max_length=500)
    limit: int = Field(default=20, ge=1, le=100)


class ContactResolveRequest(BaseModel):
    contacts: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    query: str = Field(min_length=1, max_length=500)


def _serialize(items: list[Contact]) -> list[dict[str, str]]:
    return [item.as_dict() for item in items]


@router.post("/contacts/search")
async def contacts_search(request: ContactSearchRequest) -> dict[str, object]:
    try:
        contacts = parse_contacts(request.contacts)
        matches = search_contacts(contacts, request.query, request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"contacts": _serialize(matches), "count": len(matches)}


@router.post("/contacts/resolve")
async def contacts_resolve(request: ContactResolveRequest) -> dict[str, object]:
    try:
        contacts = parse_contacts(request.contacts)
        match = resolve_contact(contacts, request.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"contact": match.as_dict() if match else None}
