"""Indoone AI core entry point.

The service has no hosted AI provider dependency. When a trained Indoone
checkpoint is present it is loaded for local inference; otherwise the local
fallback engine keeps the API runnable.
"""

from pathlib import Path

import httpx

from app.ai.inference import LocalModelRuntime
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.local_engine import LocalAIEngine
from app.ai.research import ResearchProvider, build_research_provider, format_results


MODEL_DIR = Path("models/indoone-small")
KNOWLEDGE_DIR = Path("data/knowledge")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
_fallback_engine = LocalAIEngine()
_runtime: LocalModelRuntime | None = None
_knowledge_base: LocalKnowledgeBase | None = None
_research_provider: ResearchProvider | None = build_research_provider()

if _checkpoint.exists() and _tokenizer.exists():
    _runtime = LocalModelRuntime(_checkpoint, _tokenizer)
if KNOWLEDGE_DIR.exists() and list(KNOWLEDGE_DIR.glob("*.txt")):
    _knowledge_base = LocalKnowledgeBase.from_directory(KNOWLEDGE_DIR)


_RESEARCH_TRIGGERS = (
    "latest",
    "today",
    "current",
    "currently",
    "recent",
    "news",
    "right now",
    "this week",
    "research",
    "look up",
    "search for",
)


def should_research(message: str) -> bool:
    """Return whether a message explicitly asks for fresh information."""

    normalized = " ".join(message.casefold().split())
    return any(trigger in normalized for trigger in _RESEARCH_TRIGGERS)


def _build_context(
    message: str,
    history: list[tuple[str, str]],
    knowledge: str = "",
    research: str = "",
) -> str:
    prompt_parts = ["<conversation>"]
    for role, content in history:
        prompt_parts.append(f"{role}: {content}")
    if knowledge:
        prompt_parts.append(knowledge)
    if research:
        prompt_parts.append(research)
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

        research = ""
        if _research_provider is not None and should_research(prompt):
            try:
                research = format_results(await _research_provider.search(prompt, limit=5))
            except (httpx.HTTPError, RuntimeError, ValueError):
                research = ""

        context = _build_context(
            prompt,
            history or [],
            knowledge=knowledge,
            research=research,
        )
        if _runtime is not None:
            return _runtime.generate(context)
        return await _fallback_engine.generate(context)


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    return await LocalAIService().generate(message, history=history)
