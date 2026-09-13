from __future__ import annotations

from fastapi import HTTPException, Request


def current_user_id(request: Request) -> str:
    principal = str(getattr(request.state, "principal_id", "")).strip()
    if not principal:
        raise HTTPException(status_code=401, detail="authenticated user required")
    return principal


def enforce_user_match(request: Request, requested_user_id: str) -> str:
    principal = current_user_id(request)
    requested = requested_user_id.strip()
    if requested and requested != principal:
        raise HTTPException(status_code=403, detail="user scope mismatch")
    return principal


def require_owned_file(request: Request, file_owner: str) -> str:
    principal = current_user_id(request)
    owner = file_owner.strip()
    if not owner:
        raise HTTPException(status_code=403, detail="file ownership is missing")
    if owner != principal:
        raise HTTPException(status_code=403, detail="file scope mismatch")
    return principal
