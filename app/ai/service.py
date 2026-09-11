"""Indoone AI core entry point.

The service has no hosted AI provider dependency. When a trained Indoone
checkpoint is present it is loaded for local inference; otherwise a small
local fallback keeps the chat API available until model training is complete.

When Backblaze B2 is configured, missing model artifacts are downloaded from
private object storage before local inference is initialized.
"""

from pathlib import Path
import logging
import re

import httpx

from app.ai.grounding import GroundedEvidence, append_sources, build_grounded_prompt_instruction
from app.ai.inference import LocalModelRuntime
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.local_engine import LocalAIEngine
from app.ai.research import ResearchProvider, ResearchResult, build_research_provider, format_results
from app.storage.b2 import B2StorageError, ensure_model_artifacts


logger = logging.getLogger(__name__)

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
    logger.info("Indoone model artifact check completed")
except B2StorageError as exc:
    logger.error("Indoone model artifact check failed: %s", exc)

if _checkpoint.exists() and _tokenizer.exists():
    try:
        _runtime = LocalModelRuntime(_checkpoint, _tokenizer)
        logger.info("Indoone local model runtime loaded successfully")
    except Exception as exc:
        logger.error("Indoone local model runtime failed to load: %s", exc)
        _runtime = None
else:
    logger.error(
        "Indoone local model artifacts are missing: checkpoint=%s tokenizer=%s",
        _checkpoint.exists(),
        _tokenizer.exists(),
    )
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


_SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Kannada", re.compile(r"[\u0C80-\u0CFF]")),
    ("Telugu", re.compile(r"[\u0C00-\u0C7F]")),
    ("Tamil", re.compile(r"[\u0B80-\u0BFF]")),
    ("Malayalam", re.compile(r"[\u0D00-\u0D7F]")),
    ("Hindi", re.compile(r"[\u0900-\u097F]")),
    ("Bengali", re.compile(r"[\u0980-\u09FF]")),
    ("Gujarati", re.compile(r"[\u0A80-\u0AFF]")),
    ("Punjabi", re.compile(r"[\u0A00-\u0A7F]")),
    ("Odia", re.compile(r"[\u0B00-\u0B7F]")),
)


def _detect_response_language(message: str) -> str:
    """Choose the response language from the user's message.

    Script detection is deliberately conservative: when a native Indic script
    is present, that language takes priority. Otherwise Latin-script input
    defaults to English, which avoids returning Kannada for simple messages
    such as 'Hi'.
    """

    for language, pattern in _SCRIPT_RANGES:
        if pattern.search(message):
            return language
    return "English"


def _language_instruction(language: str) -> str:
    return (
        f"Respond in {language}. Preserve the user's language/script. "
        "Do not switch to another language unless the user explicitly asks you to. "
        "For mixed-language messages, keep the same natural language mix while "
        "remaining clear and concise."
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
    response_language = _detect_response_language(message)
    prompt_parts = [
        "<conversation>",
        build_grounded_prompt_instruction(),
        f"<response_language>{response_language}</response_language>",
        _language_instruction(response_language),
    ]
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
    language = _detect_response_language(message)

    if language == "Kannada":
        if normalized in {"hi", "hello", "hey"}:
            return "ನಮಸ್ಕಾರ 👋 ನಾನು Indoone AI."
        if "namaskara" in normalized or "namaste" in normalized:
            return "ನಮಸ್ಕಾರ 👋 ನಾನು Indoone AI."
        return (
            "Indoone AI backend ಸಿದ್ಧವಾಗಿದೆ ✅, ಆದರೆ ತರಬೇತಿ ಪಡೆದ local model ಇನ್ನೂ load ಆಗಿಲ್ಲ. "
            "ಈಗ ನಾನು fallback modeನಲ್ಲಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದ್ದೇನೆ."
        )

    if language == "Telugu":
        return "నమస్కారం 👋 నేను Indoone AI. ప్రస్తుతం local trained model load కాలేదు, కాబట్టి fallback modeలో పని చేస్తున్నాను."

    if language == "Tamil":
        return "வணக்கம் 👋 நான் Indoone AI. தற்போது பயிற்சி பெற்ற local model load ஆகவில்லை; fallback mode-ல் இயங்குகிறேன்."

    if language == "Malayalam":
        return "നമസ്കാരം 👋 ഞാൻ Indoone AI. പരിശീലനം ലഭിച്ച local model ഇപ്പോൾ load ചെയ്തിട്ടില്ല; fallback mode-ൽ പ്രവർത്തിക്കുന്നു."

    if language == "Hindi":
        return "नमस्कार 👋 मैं Indoone AI हूँ। अभी trained local model load नहीं हुआ है, इसलिए मैं fallback mode में काम कर रहा हूँ।"

    if language == "Bengali":
        return "নমস্কার 👋 আমি Indoone AI। প্রশিক্ষিত local model এখনও load হয়নি, তাই আমি fallback mode-এ কাজ করছি।"

    if language == "Gujarati":
        return "નમસ્કાર 👋 હું Indoone AI છું. હાલમાં trained local model load થયું નથી, તેથી હું fallback modeમાં કામ કરી રહ્યો છું."

    if language == "Punjabi":
        return "ਸਤ ਸ੍ਰੀ ਅਕਾਲ 👋 ਮੈਂ Indoone AI ਹਾਂ। trained local model ਹਾਲੇ load ਨਹੀਂ ਹੋਇਆ, ਇਸ ਲਈ ਮੈਂ fallback mode ਵਿੱਚ ਕੰਮ ਕਰ ਰਿਹਾ ਹਾਂ।"

    if language == "Odia":
        return "ନମସ୍କାର 👋 ମୁଁ Indoone AI। trained local model ଏଯାବତ୍ load ହୋଇନାହିଁ, ସେଥିପାଇଁ ମୁଁ fallback modeରେ କାମ କରୁଛି।"

    if normalized in {"hi", "hello", "hey"}:
        return "Hi 👋 I’m Indoone AI."
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
