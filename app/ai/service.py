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


async def generate_reply(message: str) -> str:
    if _runtime is not None:
        return _runtime.generate(message)
    return await _fallback_engine.generate(message)
