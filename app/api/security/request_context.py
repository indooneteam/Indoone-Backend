from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

_REQUEST_ID: ContextVar[str] = ContextVar("indoon_request_id", default="")
_PRINCIPAL_ID: ContextVar[str] = ContextVar("indoone_principal_id", default="")


def new_request_id() -> str:
    request_id = str(uuid4())
    _REQUEST_ID.set(request_id)
    return request_id


def get_request_id() -> str:
    return _REQUEST_ID.get()


def set_principal_id(principal_id: str) -> None:
    _PRINCIPAL_ID.set(principal_id.strip())


def clear_principal_id() -> None:
    _PRINCIPAL_ID.set("")


def get_principal_id() -> str:
    return _PRINCIPAL_ID.get()


def require_principal_id() -> str:
    principal_id = get_principal_id().strip()
    if not principal_id:
        raise RuntimeError("authenticated user required")
    return principal_id
