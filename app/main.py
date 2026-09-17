from contextlib import asynccontextmanager
import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.ai import service as ai_service
from app.ai.language_detection import detect_response_language
from app.api.agent_async import router as agent_async_router
from app.api.approvals import router as approvals_router
from app.api.auth import extract_principal, validate_production_security_config
from app.api.capabilities import router as capabilities_router
from app.api.canva import router as canva_router
from app.api.chat import router as chat_router
from app.api.connector_security import enforce_connector_user_scope
from app.api.contacts import router as contacts_router
from app.api.conversations import router as conversations_router
from app.api.documents import router as documents_router
from app.api.errors import error_response
from app.api.facebook import router as facebook_router
from app.api.google_photos import router as google_photos_router
from app.api.integrations import router as integrations_router
from app.api.instagram import router as instagram_router
from app.api.memory import router as memory_router
from app.api.phone import router as phone_router
from app.api.platform import router as platform_router
from app.api.request_context import clear_principal_id, get_request_id, new_request_id, set_principal_id
from app.api.telegram import router as telegram_router
from app.api.whatsapp import router as whatsapp_router
from app.api.youtube import router as youtube_router
from app.api.voice_session import router as voice_session_router
from app.capabilities.db_runtime import configure_sqlite_runtime, sqlite_runtime_status
from app.capabilities.store import initialize as initialize_capability_store

logger = logging.getLogger("indoone.api")
ai_service._detect_response_language = detect_response_language

_DEFAULT_MAX_REQUEST_BYTES = 50 * 1024 * 1024
_PROCESS_STARTED = time.monotonic()
_REQUEST_COUNT = 0
_STATUS_COUNTS: dict[str, int] = {}


class RequestBodyTooLarge(Exception):
    pass


def _max_request_bytes() -> int:
    raw = os.getenv("INDOONE_MAX_REQUEST_BYTES", str(_DEFAULT_MAX_REQUEST_BYTES)).strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_REQUEST_BYTES
    return max(1, min(value, 100 * 1024 * 1024))


def _limited_receive(receive, max_bytes: int):
    received = 0

    async def wrapped_receive():
        nonlocal received
        message = await receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > max_bytes:
                raise RequestBodyTooLarge
        return message

    return wrapped_receive


def _error_response_with_request_id(status_code: int, code: str, message: str, request_id: str, details=None) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content=error_response(code, message, details))
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    return response


def _log_request(request: Request, request_id: str, started: float, status_code: int) -> None:
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    status_key = str(status_code)
    _STATUS_COUNTS[status_key] = _STATUS_COUNTS.get(status_key, 0) + 1
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_production_security_config()
    configure_sqlite_runtime()
    initialize_capability_store()
    yield


app = FastAPI(title="Indoone Backend", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    started = time.perf_counter()
    request_id = new_request_id()
    clear_principal_id()
    request.state.principal_id = ""
    authorization = request.headers.get("authorization", "")
    principal = ""
    if authorization:
        try:
            principal = extract_principal(authorization)
            set_principal_id(principal)
        except (RuntimeError, ValueError) as exc:
            response = _error_response_with_request_id(401, "AUTH_INVALID", str(exc), request_id)
            _log_request(request, request_id, started, response.status_code)
            return response
    elif os.getenv("INDOONE_AUTH_REQUIRED", "false").strip().lower() == "true":
        response = _error_response_with_request_id(401, "AUTH_REQUIRED", "bearer authentication required", request_id)
        _log_request(request, request_id, started, response.status_code)
        return response

    content_length = request.headers.get("content-length")
    max_request_bytes = _max_request_bytes()
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            response = _error_response_with_request_id(400, "CONTENT_LENGTH_INVALID", "invalid content-length header", request_id)
            _log_request(request, request_id, started, response.status_code)
            return response
        if declared_length < 0 or declared_length > max_request_bytes:
            response = _error_response_with_request_id(413, "REQUEST_TOO_LARGE", "request body exceeds configured size limit", request_id)
            _log_request(request, request_id, started, response.status_code)
            return response

    request._receive = _limited_receive(request._receive, max_request_bytes)

    if principal:
        request.state.principal_id = principal

    try:
        await enforce_connector_user_scope(request)
        response = await call_next(request)
    except RequestBodyTooLarge:
        response = _error_response_with_request_id(413, "REQUEST_TOO_LARGE", "request body exceeds configured size limit", request_id)
        _log_request(request, request_id, started, response.status_code)
        return response
    except HTTPException as exc:
        response = _error_response_with_request_id(exc.status_code, f"HTTP_{exc.status_code}", str(exc.detail), request_id)
        _log_request(request, request_id, started, response.status_code)
        return response

    response.headers["X-Request-ID"] = get_request_id() or request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    _log_request(request, request_id, started, response.status_code)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return _error_response_with_request_id(exc.status_code, f"HTTP_{exc.status_code}", message, get_request_id() or str(getattr(request.state, "request_id", "")))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return _error_response_with_request_id(422, "VALIDATION_ERROR", "request validation failed", get_request_id() or str(getattr(request.state, "request_id", "")), {"errors": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = get_request_id() or str(getattr(request.state, "request_id", ""))
    route = getattr(request.scope.get("route"), "path", None) or request.url.path
    logger.exception(
        "unhandled request exception",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "route": route,
            "exception_type": type(exc).__name__,
        },
    )
    return _error_response_with_request_id(500, "INTERNAL_ERROR", "internal server error", request_id)


app.include_router(chat_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(platform_router, prefix="/api")
app.include_router(agent_async_router, prefix="/api")
app.include_router(memory_router, prefix="/api")
app.include_router(capabilities_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(voice_session_router, prefix="/api")
app.include_router(contacts_router, prefix="/api")
app.include_router(phone_router, prefix="/api")
app.include_router(approvals_router, prefix="/api")
app.include_router(integrations_router, prefix="/api")
app.include_router(telegram_router, prefix="/api")
app.include_router(whatsapp_router, prefix="/api")
app.include_router(youtube_router, prefix="/api")
app.include_router(instagram_router, prefix="/api")
app.include_router(facebook_router, prefix="/api")
app.include_router(google_photos_router, prefix="/api")
app.include_router(canva_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    try:
        sqlite_runtime_status()
    except Exception:
        logger.exception("readiness check failed")
        return _error_response_with_request_id(503, "NOT_READY", "service dependencies are not ready", get_request_id() or "")
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/health/details")
async def health_details() -> dict[str, object]:
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _PROCESS_STARTED, 2),
        "requests_total": _REQUEST_COUNT,
        "responses_by_status": dict(sorted(_STATUS_COUNTS.items())),
        "sqlite": sqlite_runtime_status(),
    }
