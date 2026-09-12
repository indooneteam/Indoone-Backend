from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

_REQUEST_ID: ContextVar[str] = ContextVar("indoon_request_id", default="")


def new_request_id() -> str:
    request_id = str(uuid4())
    _REQUEST_ID.set(request_id)
    return request_id


def get_request_id() -> str:
    return _REQUEST_ID.get()
