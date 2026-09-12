from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.ai import service as ai_service
from app.ai.language_detection import detect_response_language
from app.api.auth import extract_principal
from app.api.capabilities import router as capabilities_router
from app.api.canva import router as canva_router
from app.api.chat import router as chat_router
from app.api.contacts import router as contacts_router
from app.api.documents import router as documents_router
from app.api.errors import error_response
from app.api.facebook import router as facebook_router
from app.api.google_photos import router as google_photos_router
from app.api.integrations import router as integrations_router
from app.api.instagram import router as instagram_router
from app.api.phone import router as phone_router
from app.api.platform import router as platform_router
from app.api.request_context import get_request_id, new_request_id, set_principal_id
from app.api.telegram import router as telegram_router
from app.api.whatsapp import router as whatsapp_router
from app.api.youtube import router as youtube_router
from app.api.voice_session import router as voice_session_router
from app.capabilities.store import initialize as initialize_capability_store

# Keep one canonical detector for production chat requests.
ai_service._detect_response_language = detect_response_language


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_capability_store()
    yield


app = FastAPI(title="Indoone Backend", version="0.3.0", lifespan=lifespan)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = new_request_id()
    authorization = request.headers.get("authorization", "")
    principal = ""
    if authorization:
        try:
            principal = extract_principal(authorization)
            set_principal_id(principal)
        except (RuntimeError, ValueError) as exc:
            return JSONResponse(status_code=401, content=error_response("AUTH_INVALID", str(exc)))
    elif os.getenv("INDOONE_AUTH_REQUIRED", "false").strip().lower() == "true":
        return JSONResponse(status_code=401, content=error_response("AUTH_REQUIRED", "bearer authentication required"))

    if principal:
        request.state.principal_id = principal
    response = await call_next(request)
    response.headers["X-Request-ID"] = get_request_id() or request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return JSONResponse(status_code=exc.status_code, content=error_response(f"HTTP_{exc.status_code}", message))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=error_response("VALIDATION_ERROR", "request validation failed", {"errors": exc.errors()}))


app.include_router(chat_router, prefix="/api")
app.include_router(platform_router, prefix="/api")
app.include_router(capabilities_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(voice_session_router, prefix="/api")
app.include_router(contacts_router, prefix="/api")
app.include_router(phone_router, prefix="/api")
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
