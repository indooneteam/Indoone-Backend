from fastapi import FastAPI

app = FastAPI(title="Indoone Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "indoone-backend"}


@app.post("/api/v1/chat")
def chat(message: dict[str, str]) -> dict[str, str]:
    text = message.get("message", "").strip()
    if not text:
        return {"reply": "Please enter a message."}

    return {
        "reply": f"Indoone received: {text}",
    }
