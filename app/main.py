from fastapi import FastAPI

from app.ai import service as ai_service
from app.ai.language_detection import detect_response_language

# Keep one canonical detector for production chat requests.
ai_service._detect_response_language = detect_response_language

from app.api.chat import router as chat_router

app = FastAPI(title="Indoone Backend", version="0.1.0")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
