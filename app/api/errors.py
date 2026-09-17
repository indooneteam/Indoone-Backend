from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.api.request_context import get_request_id


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None


def error_response(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return asdict(ErrorEnvelope(code=code, message=message, request_id=get_request_id(), details=details))
