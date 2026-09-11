"""Indoone AI core entry point.

The service has no hosted AI provider dependency. When a trained Indoone
checkpoint is present it is loaded for local inference; otherwise a small
local fallback keeps the chat API available until model training is complete.

When Backblaze B2 is configured, missing model artifacts are downloaded from
private object storage before local inference is initialized.
"""

from pathlib import Path

import httpx

from app.ai.grounding import GroundedEvidence, append_sources, build_grounded_prompt_instruction
from app.ai.inference import LocalModelRuntime
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.local_engine import LocalAIEngine
from app.ai.research import ResearchProvider, ResearchResult, build_research_provider, format_results
from app.storage.b2 import B2StorageError, ensure_model_artifacts


MODEL_DIR = Path("models/indoone-small")
KNOWLEDGE_DIR = Path("data/knowledge")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
_fallback_engine = LocalAIEngine()
_runtime: LocalModelRuntime | None = None
_knowledge_base: LocalKnowledgeBase | None = None
_research_provider: ResearchProvider | None = build_research_provider()

try:
    ensure_model_artifacts(MODEL_DIR)
except B2StorageError:
    # B2 is an optional model-artifact source; local fallback remains available.
    pass

if _checkpoint.exists() and _tokenizer.exists():
    try:
        _runtime = LocalModelRuntime(_checkpoint, _tokenizer)
    except Exception:
        _runtime = None
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
    prompt_parts = ["<conversation>", build_grounded_prompt_instruction()]
    for role, content in history:
        prompt_parts.append(f"{role}: {content}")
    if knowledge:
        prompt_parts.append(knowledge)
    if research:
        prompt_parts.append(research)
    prompt_parts.append(f"user: {message.strip()}")
    prompt_parts.append("assistant:")
    return "\n".join(prompt_parts)


def _evidence_from_results(results: list[ResearchResult]) -> list[GroundedEvidence]:
    return [GroundedEvidence(title=result.title, url=result.url, snippet=result.snippet) for result in results]


def _fallback_reply(message: str) -> str:
    """Return a truthful local response while the trained checkpoint is absent."""

    normalized = " ".join(message.casefold().split())
    if normalized in {"hi", "hello", "hey", "namaskara", "namaste"}:
        return (
            "Namaskara 👋 I’m Indoone AI. The production backend is online. "
            "My trained local model is not loaded yet, so I’m currently running in fallback mode."
        )
    if "what is indoone" in normalized or "indoone ai" in normalized:
        return (
            "Indoone AI is the AI service behind the Indoone app. The backend is online, "
            "but the first trained Indoone local checkpoint still needs to be produced."
        )
    return (
        "Indoone AI backend is reachable ✅, but the trained Indoone local model is not "
        "available yet. I’m keeping the API online in fallback mode instead of returning "
        "an error. Full AI generation will start after the local checkpoint is trained and deployed."
    )


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
        research_results: list[ResearchResult] = []
        if _research_provider is not None and should_research(prompt):
            try:
                research_results = await _research_provider.search(prompt, limit=5)
                research = format_results(research_results)
            except (httpx.HTTPError, RuntimeError, ValueError):
                research_results = []
                research = ""

        context = _build_context(
            prompt,
            history or [],
            knowledge=knowledge,
            research=research,
        )
        if _runtime is not None:
            try:
                answer = _runtime.generate(context)
            except RuntimeError:
                answer = _fallback_reply(prompt)
        elif getattr(_fallback_engine, "ready", True):
            try:
                answer = await _fallback_engine.generate(context)
            except RuntimeError:
                answer = _fallback_reply(prompt)
        else:
            answer = _fallback_reply(prompt)
        return append_sources(answer, _evidence_from_results(research_results))


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    return await LocalAIService().generate(message, history=history)
