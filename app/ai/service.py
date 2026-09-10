"""Indoone AI core entry point.

The service has no hosted AI provider dependency. When a trained Indoone
checkpoint is present it is loaded for local inference; otherwise the local
fallback engine keeps the API runnable.
"""

from pathlib import Path

from app.ai.inference import LocalModelRuntime
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.local_engine import LocalAIEngine


MODEL_DIR = Path("models/indoone-small")
KNOWLEDGE_DIR = Path("data/knowledge")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
_fallback_engine = LocalAIEngine()
_runtime: LocalModelRuntime | None = None
_knowledge_base: LocalKnowledgeBase | None = None

if _checkpoint.exists() and _tokenizer.exists():
    _runtime = LocalModelRuntime(_checkpoint, _tokenizer)
if KNOWLEDGE_DIR.exists() and list(KNOWLEDGE_DIR.glob("*.txt")):
    _knowledge_base = LocalKnowledgeBase.from_directory(KNOWLEDGE_DIR)


def _build_context(
    message: str,
    history: list[tuple[str, str]],
    knowledge: str = "",
) -> str:
    prompt_parts = ["<conversation>"]
    for role, content in history:
        prompt_parts.append(f"{role}: {content}")
    if knowledge:
        prompt_parts.append(knowledge)
    prompt_parts.append(f"user: {message.strip()}")
    prompt_parts.append("assistant:")
    return "\n".join(prompt_parts)


class LocalAIService:
    """Async service facade for the Indoone local AI runtime."""

    async def generate(self, message: str, history: list[tuple[str, str]] | None = None) -> str:
        prompt = message.strip()
        if not prompt:
            raise ValueError("message cannot be empty")

        knowledge = ""
        if _knowledge_base is not None:
            knowledge = format_hits(_knowledge_base.search(prompt, limit=3))
        context = _build_context(prompt, history or [], knowledge=knowledge)
        if _runtime is not None:
            return _runtime.generate(context)
        return await _fallback_engine.generate(context)


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    return await LocalAIService().generate(message, history=history)
