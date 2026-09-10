"""Indoone AI core entry point.

The service has no hosted AI provider dependency. When a trained Indoone
checkpoint is present it is loaded for local inference; until then the engine
returns a deterministic development response so the API remains runnable.
"""

from pathlib import Path

from app.ai.inference import LocalModelRuntime
from app.ai.local_engine import LocalAIEngine


MODEL_DIR = Path("models/indoone-small")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
_fallback_engine = LocalAIEngine()
_runtime: LocalModelRuntime | None = None

if _checkpoint.exists() and _tokenizer.exists():
    _runtime = LocalModelRuntime(_checkpoint, _tokenizer)


class LocalAIService:
    """Async service facade for the Indoone local AI runtime."""

    async def generate(self, message: str) -> str:
        prompt = message.strip()
        if not prompt:
            raise ValueError("message cannot be empty")
        if _runtime is not None:
            return _runtime.generate(prompt)
        return await _fallback_engine.generate(prompt)


async def generate_reply(message: str) -> str:
    return await LocalAIService().generate(message)
