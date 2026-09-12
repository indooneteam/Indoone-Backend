from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.ai import service as ai_service
from app.ai.language_detection import detect_response_language
from app.api.capabilities import router as capabilities_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.platform import router as platform_router
from app.api.voice_session import router as voice_session_router
from app.capabilities.store import initialize as initialize_capability_store

# Keep one canonical detector for production chat requests.
ai_service._detect_response_language = detect_response_language


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_capability_store()
    yield


app = FastAPI(title="Indoone Backend", version="0.3.0", lifespan=lifespan)
app.include_router(chat_router, prefix="/api")
app.include_router(platform_router, prefix="/api")
app.include_router(capabilities_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(voice_session_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
