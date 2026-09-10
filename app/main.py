from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(title="Indoone Backend", version="0.1.0")
app.include_router(chat_router, prefix="/api")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
