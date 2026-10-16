"""One shared, Indoone-branded failure message, plus the helper that logs the real cause.

Provider routes used to put the provider's own error text straight into the response
detail, and the app shows that detail to the user. This module is the single place that
decides what the user is allowed to read.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import HTTPException

from app.api.request_context import get_request_id

logger = logging.getLogger("indoone.errors")


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None


def error_response(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return asdict(ErrorEnvelope(code=code, message=message, request_id=get_request_id(), details=details))


# What the user is told when a request fails on our side or on a provider's side. It names
# Indoone, says what happened in one sentence, and reveals nothing about the provider, the
# operation or our configuration. Every route returns this same wording, so the app reads
# the same way whatever broke.
REQUEST_FAILED_MESSAGE = (
    "Indoone could not complete this request. Please try again in a moment."
)


def provider_failure(
    operation: str,
    exc: BaseException,
    status_code: int = 502,
) -> HTTPException:
    """Log the real failure and return the error the client is allowed to see.

    The provider's own text stays in the log, where it is useful. What leaves here names
    Indoone and nothing else, because the app shows this message to the user as our own
    words.

    The status code is passed through unchanged: the app retries 502, 503 and 504, so
    collapsing them would change behaviour.
    """
    logger.error("%s failed: %s", operation, exc, exc_info=True)
    return HTTPException(status_code=status_code, detail=REQUEST_FAILED_MESSAGE)
