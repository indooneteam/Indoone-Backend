"""Indoone AI core entry point.

The service keeps inference local and provider-independent. A trained Indoone
checkpoint is used when available; otherwise the small local fallback keeps
the chat API available.
"""

from datetime import datetime, timedelta, timezone
import asyncio
from pathlib import Path
import logging
import threading
from typing import TYPE_CHECKING
import re
import time
from urllib.parse import urlparse

import httpx

from app.ai.answer_quality import assess_answer, user_safe_failure
from app.ai.general_knowledge import WikipediaKnowledgeProvider, is_general_knowledge_question
from app.ai.grounding import (
    GroundedEvidence,
    append_sources,
    build_grounded_prompt_instruction,
)
from app.ai.intent import classify_intent
from app.ai.question_understanding import understand_question
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.training_data import format_instruction_prompt
from app.ai.research import (
    ResearchProvider,
    ResearchResult,
    build_research_provider,
    format_research_context,
    format_results,
)
from app.storage.b2 import B2StorageError, ensure_model_artifacts
from app.storage.github_release import GitHubReleaseStorageError, get_github_release_storage


logger = logging.getLogger(__name__)

_general_knowledge_provider = WikipediaKnowledgeProvider()

if TYPE_CHECKING:
    from app.ai.inference import LocalModelRuntime

# Kept as a lazy injection point for tests and compatibility. The real class is
# imported only inside _load_local_model_runtime() so Render startup stays light.
LocalModelRuntime = None

class _LightweightFallbackEngine:
    """Async fallback adapter that never loads the trained model."""
    async def generate(self, message: str) -> str:
        # The production caller passes a full context. Recover the final user
        # message when possible so the greeting fallback remains natural.
        prompt = message.strip()
        marker = re.search(r"\nuser:\s*(.+?)\s*</instruction>\s*\Z", prompt, flags=re.IGNORECASE | re.DOTALL)
        if marker:
            prompt = marker.group(1).strip()
        return _fallback_reply(prompt)


# Compatibility injection point for tests and lightweight fallback adapters.
_fallback_engine = _LightweightFallbackEngine()

MODEL_DIR = Path("models/indoone-small")
KNOWLEDGE_DIR = Path("data/knowledge")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
# Keep model loading entirely request-driven. Constructing LocalAIEngine at import
# time can eagerly load the full PyTorch checkpoint during Render startup, which
# increases memory pressure and can cause the web process to restart before chat.
_runtime: "LocalModelRuntime | None" = None
_knowledge_base: LocalKnowledgeBase | None = None
_research_provider: ResearchProvider | None = build_research_provider()
_NEXT_MODEL_LOAD_ATTEMPT = 0.0
_MODEL_LOAD_RETRY_SECONDS = 60.0
_MODEL_GENERATION_TIMEOUT_SECONDS = 20.0
_MODEL_MAX_NEW_TOKENS = 128

_CREATIVE_MARKERS = (
    "write a ",
    "write an ",
    "story",
    "poem",
    "joke",
    "funny",
    "creative",
    "dialogue",
    "roleplay",
    "brainstorm",
)
# Serialize the artifact check/runtime load so concurrent chat requests cannot
# trigger duplicate B2 downloads or duplicate model loads.
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_INFERENCE_LOCK = threading.Lock()
_ARTIFACT_CHECK_COMPLETED = False


def _ensure_model_artifacts(model_dir: Path) -> None:
    """Load model artifacts from the configured private GitHub Release, else B2."""
    github_storage = get_github_release_storage()
    if github_storage is None:
        ensure_model_artifacts(model_dir)
        return

    artifacts = {
        "indoone-small.pt": model_dir / "indoone-small.pt",
        "tokenizer.json": model_dir / "tokenizer.json",
    }
    for filename, local_path in artifacts.items():
        if local_path.exists():
            continue
        github_storage.download_file(filename, local_path)


def _next_utc_midnight_timestamp() -> float:
    """Back off failed remote model-artifact downloads until the next UTC day."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date()
    midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return midnight.timestamp()


def _load_local_model_runtime() -> "LocalModelRuntime | None":
    """Load the trained checkpoint with a single-flight artifact download/load."""
    global _runtime, _NEXT_MODEL_LOAD_ATTEMPT, _ARTIFACT_CHECK_COMPLETED

    if _runtime is not None:
        return _runtime

    if not _MODEL_LOAD_LOCK.acquire(blocking=False):
        logger.info("Indoone model load already in progress; using fallback for this request")
        return None

    try:
        # A concurrent request may have completed the model load while this
        # request was waiting for the lock.
        if _runtime is not None:
            return _runtime

        now = time.time()
        if now < _NEXT_MODEL_LOAD_ATTEMPT:
            return None

        if not _ARTIFACT_CHECK_COMPLETED:
            try:
                _ensure_model_artifacts(MODEL_DIR)
                if get_github_release_storage() is not None:
                    logger.info("Indoone model artifacts checked from private GitHub Release")
                else:
                    logger.info("Indoone model artifact check completed from B2")
            except (B2StorageError, GitHubReleaseStorageError) as exc:
                _NEXT_MODEL_LOAD_ATTEMPT = _next_utc_midnight_timestamp()
                logger.error("Indoone model artifact check failed: %s", exc)
                return None

            if not (_checkpoint.exists() and _tokenizer.exists()):
                _NEXT_MODEL_LOAD_ATTEMPT = _next_utc_midnight_timestamp()
                logger.error(
                    "Indoone local model artifacts are missing: checkpoint=%s tokenizer=%s",
                    _checkpoint.exists(), _tokenizer.exists(),
                )
                return None

            # From this point on, every request in this process uses the local
            # artifacts and never re-checks/downloads them from remote storage.
            _ARTIFACT_CHECK_COMPLETED = True

        try:
            # Import PyTorch/model code only when a chat request actually needs
            # trained-model inference. This keeps Render startup memory low while
            # retaining a testable lazy injection point.
            runtime_factory = LocalModelRuntime
            if runtime_factory is None:
                from app.ai.inference import LocalModelRuntime as runtime_factory

            _runtime = runtime_factory(_checkpoint, _tokenizer)
            _NEXT_MODEL_LOAD_ATTEMPT = 0.0
            logger.info("Indoone local model runtime loaded successfully")
        except Exception as exc:
            _NEXT_MODEL_LOAD_ATTEMPT = now + _MODEL_LOAD_RETRY_SECONDS
            logger.error("Indoone local model runtime failed to load: %s", exc)
            _runtime = None
        return _runtime
    finally:
        _MODEL_LOAD_LOCK.release()


# Model loading is lazy: deployments and health checks must not consume remote model-storage bandwidth.
if KNOWLEDGE_DIR.exists() and list(KNOWLEDGE_DIR.glob("*.txt")):
    _knowledge_base = LocalKnowledgeBase.from_directory(KNOWLEDGE_DIR)


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
    ("Urdu", re.compile(r"[\u0600-\u06FF]")),
)

_LANGUAGE_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("kannada", "ಕನ್ನಡ", "ಕನ್ನಡದ")),
    ("Hindi", ("hindi", "हिंदी", "हिन्दी")),
    ("Telugu", ("telugu", "తెలుగు")),
    ("Tamil", ("tamil", "தமிழ்", "தமிழ")),
    ("Malayalam", ("malayalam", "മലയാളം")),
    ("Marathi", ("marathi", "मराठी")),
    ("Bengali", ("bengali", "bangla", "বাংলা", "বাঙালি")),
    ("Assamese", ("assamese", "অসমীয়া", "অসমিয়া")),
    ("Gujarati", ("gujarati", "ગુજરાતી")),
    ("Punjabi", ("punjabi", "ਪੰਜਾਬੀ")),
    ("Odia", ("odia", "oriya", "ଓଡ଼ିଆ", "ଓଡିଆ")),
    ("Urdu", ("urdu", "اردو")),
)

_ROMANIZED_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("bagge", "helu", "heLi", "maadu", "maadi", "madbeku", "madbekagutte", "enidu", "enu", "yenu", "yenide", "ivaga", "matte", "nanage", "nimage", "nanna", "namma", "ide", "illa", "agide", "beku")),
    ("Hindi", ("kya", "hai", "hain", "mujhe", "aap", "aapko", "batao", "bataiye", "kaise", "kaisa", "kahan", "kyun", "nahi", "abhi", "mera", "meri")),
    ("Telugu", ("enti", "ela", "cheppu", "cheppandi", "undi", "ledu", "nenu", "meeru", "naaku", "emiti", "enduku", "ippudu", "malli")),
    ("Tamil", ("enna", "epdi", "eppadi", "sollu", "sollunga", "irukku", "illa", "naan", "neenga", "enakku", "ippo", "yen")),
    ("Malayalam", ("entha", "engane", "parayu", "parayoo", "undu", "illa", "njan", "ningal", "enikku", "ippo")),
    ("Marathi", ("kay", "aahe", "mala", "tumhi", "sanga", "kasa", "kashi", "kuthe", "ka", "nahi", "aata")),
    ("Bengali", ("ki", "ache", "ami", "apni", "bolo", "bolun", "amar", "keno", "nei")),
    ("Assamese", ("ki", "ase", "moi", "tumi", "kobo", "mur", "kio", "nai")),
    ("Gujarati", ("shu", "che", "chhe", "mane", "tame", "kaho", "kem", "nathi", "maru")),
    ("Punjabi", ("menu", "mainu", "tusi", "daso", "kive", "kiwe", "nahi", "mera", "sanu")),
    ("Odia", ("kana", "achhi", "mu", "tame", "kahantu", "kemiti", "nahi", "ebe")),
    ("Urdu", ("kya", "hai", "mujhe", "aap", "batao", "bataiye", "kaise", "kyun", "nahi")),
)

_RESPONSE_TAG_RE = re.compile(
    r"</?(?:instruction|response|conversation|grounding|response_language)>|<response_language>.*?</response_language>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _detect_response_language(message: str) -> str:
    normalized = " ".join(message.casefold().split())
    for language, names in _LANGUAGE_NAMES:
        for name in names:
            candidate = name.casefold()
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized):
                return language
    for language, pattern in _SCRIPT_RANGES:
        if pattern.search(message):
            return language
    scores: dict[str, int] = {}
    for language, hints in _ROMANIZED_HINTS:
        score = sum(
            1
            for hint in hints
            if re.search(rf"(?<!\w){re.escape(hint.casefold())}(?!\w)", normalized)
        )
        if score:
            scores[language] = score
    if scores:
        best_language, best_score = max(scores.items(), key=lambda item: item[1])
        tied = [language for language, score in scores.items() if score == best_score]
        if best_score >= 2 or len(tied) == 1:
            return best_language
    return "English"


def _is_fast_fallback_message(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return normalized in {"hi", "hello", "hey", "namaskara", "namaste"}


def _is_learning_request(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return (
        "teach me" in normalized
        or "teach me something" in normalized
        or "learn something new" in normalized
        or "learn me" in normalized
        or "ಹೊಸ ವಿಷಯ" in normalized
        or "ಹೊಸದೇನಾದರೂ" in normalized
        or "ಕಲಿಸು" in normalized
        or "ಕಲಿಸಿ" in normalized
    )


def _learning_fallback_reply(language: str) -> str:
    if language == "Kannada":
        return "ಒಂದು ಹೊಸ ವಿಷಯ ಕಲಿಯೋಣ: ಆಕ್ಟೋಪಸ್‌ಗೆ ಮೂರು ಹೃದಯಗಳಿವೆ. ಎರಡು ಹೃದಯಗಳು ಕಿವಿರುಗಳಿಗೆ ರಕ್ತ ಕಳುಹಿಸುತ್ತವೆ ಮತ್ತು ಮೂರನೇ ಹೃದಯ ದೇಹದ ಉಳಿದ ಭಾಗಕ್ಕೆ ರಕ್ತ ಪಂಪ್ ಮಾಡುತ್ತದೆ."
    if language == "Hindi":
        return "एक नया तथ्य सीखें: ऑक्टोपस के तीन दिल होते हैं। दो दिल गलफड़ों तक रक्त पहुँचाते हैं और तीसरा दिल शरीर के बाकी हिस्से में रक्त पंप करता है."
    if language == "Telugu":
        return "ఒక కొత్త విషయం నేర్చుకుందాం: ఆక్టోపస్‌కు మూడు గుండెలు ఉంటాయి. రెండు గుండెలు గిల్లులకు రక్తాన్ని పంపుతాయి, మూడవది శరీరంలోని మిగతా భాగాలకు రక్తాన్ని పంప్ చేస్తుంది."
    if language == "Tamil":
        return "ஒரு புதிய விஷயம் கற்போம்: ஆக்டோபஸுக்கு மூன்று இதயங்கள் உள்ளன. இரண்டு இதயங்கள் கிளவுகளுக்கு இரத்தத்தை அனுப்புகின்றன; மூன்றாவது இதயம் உடலின் பிற பகுதிகளுக்கு இரத்தத்தை பம்ப் செய்கிறது."
    return "Here is something new to learn: an octopus has three hearts. Two send blood to the gills, while the third pumps blood to the rest of the body."


def _generate_with_local_model(
    runtime: "LocalModelRuntime",
    context: str,
    language: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    # Render Free provides a very small CPU budget. Serialize local inference so
    # concurrent requests cannot multiply the model's CPU/memory pressure.
    with _MODEL_INFERENCE_LOCK:
        return runtime.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            language=language,
        )


def _is_creative_request(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return any(
        normalized.startswith(marker) or f" {marker}" in normalized
        for marker in _CREATIVE_MARKERS
    )


def _generation_profile(message: str, intent_name: str) -> tuple[tuple[float, ...], int]:
    if _is_creative_request(message):
        return (0.7, 0.2), 160
    if intent_name in {"coding", "translation", "summarization", "file_qa"}:
        return (0.0, 0.15), _MODEL_MAX_NEW_TOKENS
    return (0.0, 0.2), _MODEL_MAX_NEW_TOKENS


def _language_instruction(language: str) -> str:
    return f"Respond only in {language}. Preserve the user's language and script. Do not switch languages unless the user explicitly requests it. Keep the answer natural, clear, and concise."


_KNOWLEDGE_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "the", "there", "to", "was", "were", "what", "when", "where", "who", "why",
}


def _knowledge_fallback_sentence(
    message: str,
    hits: list[object],
    *,
    minimum_score: float = 0.20,
) -> str | None:
    """Return one strongly matched local-knowledge sentence as a safe fallback."""
    if not hits:
        return None

    top_hit = hits[0]
    score = float(getattr(top_hit, "score", 0.0))
    content = str(getattr(top_hit, "content", "")).strip()
    if score < minimum_score or not content:
        return None

    query_terms = {
        token.casefold()
        for token in re.findall(r"[\w'-]+", message, flags=re.UNICODE)
        if len(token) > 1 and token.casefold() not in _KNOWLEDGE_QUERY_STOPWORDS
    }
    if not query_terms:
        return None

    candidates = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", content)
        if sentence.strip()
    ]
    best_sentence = ""
    best_overlap = 0
    for sentence in candidates:
        sentence_terms = {
            token.casefold()
            for token in re.findall(r"[\w'-]+", sentence, flags=re.UNICODE)
            if len(token) > 1
        }
        overlap = len(query_terms & sentence_terms)
        if overlap > best_overlap:
            best_sentence = sentence
            best_overlap = overlap

    if best_overlap < 2 or len(query_terms) < 2:
        return None

    effective_coverage = best_overlap / len(query_terms)
    if effective_coverage < 0.80:
        return None

    return best_sentence


def _build_context(message: str, history: list[tuple[str, str]], knowledge: str = "", research: str = "") -> str:
    response_language = _detect_response_language(message)
    grounding_instruction = build_grounded_prompt_instruction().replace("<grounding>", "").replace("</grounding>", "").strip()
    prompt_parts = [grounding_instruction, _language_instruction(response_language)]
    if history:
        prompt_parts.append("Conversation context:")
        for role, content in history:
            prompt_parts.append(f"{role}: {content}")
    if knowledge:
        prompt_parts.append("Relevant knowledge:")
        prompt_parts.append(knowledge)
    if research:
        prompt_parts.append("Fresh research evidence:")
        prompt_parts.append(research)
    prompt_parts.append(f"user: {message.strip()}")
    return format_instruction_prompt("\n".join(part for part in prompt_parts if part.strip()))


def _evidence_from_results(results: list[ResearchResult]) -> list[GroundedEvidence]:
    return [GroundedEvidence(title=result.title, url=result.url, snippet=result.snippet) for result in results]


def _research_has_enough_sources(results: list[ResearchResult], cross_check: bool) -> bool:
    unique_domains = {
        urlparse(result.url).netloc.casefold()
        for result in results
        if urlparse(result.url).netloc
    }
    return len(unique_domains) >= (3 if cross_check else 2)


def _clean_model_reply(answer: str) -> str:
    text = answer.strip()
    if not text:
        return ""
    response_start = re.search(r"<response>\s*", text, flags=re.IGNORECASE)
    if response_start:
        text = text[response_start.end():]
    instruction_end = re.search(r"</instruction>\s*", text, flags=re.IGNORECASE)
    if instruction_end:
        text = text[instruction_end.end():]
    response_end = re.search(r"</response>", text, flags=re.IGNORECASE)
    if response_end:
        text = text[:response_end.start()]
    text = _RESPONSE_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_romanized_request(message: str, language: str) -> bool:
    if language == "English":
        return False
    if any(pattern.search(message) for _, pattern in _SCRIPT_RANGES):
        return False
    normalized = " ".join(message.casefold().split())
    hints = dict(_ROMANIZED_HINTS).get(language, ())
    return any(
        re.search(rf"(?<!\w){re.escape(hint.casefold())}(?!\w)", normalized)
        for hint in hints
    )


def _has_expected_script(
    text: str,
    language: str,
    *,
    allow_romanized: bool = False,
) -> bool:
    if not text:
        return False
    if language == "English":
        return bool(re.search(r"[A-Za-z]", text))
    pattern = dict(_SCRIPT_RANGES).get(language)
    if pattern is not None and pattern.search(text):
        return True
    if allow_romanized:
        return bool(re.search(r"[A-Za-z]", text))
    return False


def _research_extract_fallback(results: list[ResearchResult]) -> str | None:
    """Return concise source evidence when synthesis is unavailable."""
    snippets: list[str] = []
    for result in results:
        snippet = " ".join(result.snippet.split()).strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet[:1200])
        if len(snippets) >= 2:
            break
    if not snippets:
        return None
    return " ".join(snippets)


def _generation_error_reply(language: str) -> str:
    messages = {
        "Kannada": "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಈಗ ಸರಿಯಾದ ಉತ್ತರವನ್ನು ರಚಿಸಲು ನನಗೆ ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಕೇಳಿ.",
        "Telugu": "క్షమించండి, ఈ ప్రశ్నకు ఇప్పుడు సరైన సమాధానం ఇవ్వలేకపోయాను. దయచేసి మళ్లీ అడగండి.",
        "Tamil": "மன்னிக்கவும், இந்தக் கேள்விக்கு இப்போது சரியான பதிலை உருவாக்க முடியவில்லை. தயவுசெய்து மீண்டும் கேளுங்கள்.",
        "Malayalam": "ക്ഷമിക്കണം, ഈ ചോദ്യത്തിന് ഇപ്പോൾ ശരിയായ മറുപടി നൽകാൻ കഴിഞ്ഞില്ല. ദയവായി വീണ്ടും ചോദിക്കൂ.",
        "Hindi": "माफ़ कीजिए, मैं अभी इस सवाल का सही जवाब तैयार नहीं कर पाया। कृपया फिर से पूछें।",
        "Bengali": "দুঃখিত, এই প্রশ্নের সঠিক উত্তর এখন দিতে পারিনি। অনুগ্রহ করে আবার জিজ্ঞাসা করুন।",
        "Gujarati": "માફ કરશો, હું હાલમાં આ પ્રશ્નનો યોગ્ય જવાબ આપી શક્યો નથી. કૃપા કરીને ફરી પૂછો.",
        "Punjabi": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਇਸ ਸਵਾਲ ਦਾ ਸਹੀ ਜਵਾਬ ਹੁਣ ਨਹੀਂ ਦੇ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਪੁੱਛੋ।",
        "Odia": "ଦୁଃଖିତ, ମୁଁ ଏହି ପ୍ରଶ୍ନର ସଠିକ ଉତ୍ତର ଏବେ ଦେଇପାରିଲି ନାହିଁ। ଦୟାକରି ପୁଣି ପଚାରନ୍ତୁ.",
        "Urdu": "معذرت، میں ابھی اس سوال کا درست جواب نہیں دے سکا۔ براہِ کرم دوبارہ پوچھیں۔",
    }
    return messages.get(language, "Sorry, I could not generate a reliable answer right now.")


def _fallback_reply(message: str) -> str:
    normalized = " ".join(message.casefold().split())
    language = _detect_response_language(message)
    if language == "Kannada":
        if normalized in {"hi", "hello", "hey"} or "namaskara" in normalized or "namaste" in normalized:
            return "ನಮಸ್ಕಾರ 👋 ನಾನು Indoone AI."
        return "Indoone AI backend ಸಿದ್ಧವಾಗಿದೆ ✅, ಆದರೆ ತರಬೇತಿ ಪಡೆದ local model ಇನ್ನೂ load ಆಗಿಲ್ಲ."
    if language == "Telugu":
        return "నమస్కారం 👋 నేను Indoone AI. ప్రస్తుతం local trained model load కాలేదు."
    if language == "Tamil":
        return "வணக்கம் 👋 நான் Indoone AI. தற்போது பயிற்சி பெற்ற local model load ஆகவில்லை."
    if language == "Malayalam":
        return "നമസ്കാരം 👋 ഞാൻ Indoone AI. പരിശീലനം ലഭിച്ച local model ഇപ്പോൾ load ചെയ്തിട്ടില്ല."
    if language == "Hindi":
        return "नमस्कार 👋 मैं Indoone AI हूँ। अभी trained local model load नहीं हुआ है।"
    if language == "Bengali":
        return "নমস্কার 👋 আমি Indoone AI। প্রশিক্ষিত local model এখনও load হয়নি।"
    if language == "Gujarati":
        return "નમસ્કાર 👋 હું Indoone AI છું. હાલમાં trained local model load થયું નથી."
    if language == "Punjabi":
        return "ਸਤ ਸ੍ਰੀ ਅਕਾਲ 👋 ਮੈਂ Indoone AI ਹਾਂ। trained local model ਹਾਲੇ load ਨਹੀਂ ਹੋਇਆ।"
    if language == "Odia":
        return "ନମସ୍କାର 👋 ମୁଁ Indoone AI। trained local model ଏଯାବତ୍ load ହୋଇନାହିଁ."
    if language == "Urdu":
        return "السلام علیکم 👋 میں Indoone AI ہوں۔ ابھی trained local model load نہیں ہوا۔"
    if normalized in {"hi", "hello", "hey"}:
        return "Hi 👋 I’m Indoone AI."
    return "Indoone AI backend is reachable ✅, but the trained Indoone local model is not available yet."


class LocalAIService:
    async def generate(
        self,
        message: str,
        history: list[tuple[str, str]] | None = None,
        document_context: str = "",
    ) -> str:
        prompt = message.strip()
        if not prompt:
            raise ValueError("message cannot be empty")

        understanding = understand_question(prompt)
        intent = classify_intent(prompt)
        knowledge_query = understanding.research_query or prompt
        knowledge = ""
        knowledge_hits = []
        if _knowledge_base is not None:
            knowledge_hits = _knowledge_base.search(knowledge_query, limit=3)
            knowledge = format_hits(knowledge_hits)
        if document_context.strip():
            knowledge = (knowledge + "\n\n" if knowledge else "") + "User-provided document:\n" + document_context.strip()[:100_000]

        research = ""
        research_results: list[ResearchResult] = []
        research_blocked = False
        if intent.needs_research:
            if _research_provider is None:
                research_blocked = True
                logger.warning("Research provider unavailable for query: %s", prompt)
            else:
                try:
                    candidate_results = await _research_provider.search(prompt, limit=8)
                    if _research_has_enough_sources(candidate_results, intent.needs_cross_check):
                        research_results = candidate_results
                        research = format_research_context(candidate_results)
                    else:
                        research_blocked = True
                        if candidate_results:
                            logger.warning("Insufficient independent research sources for query: %s", prompt)
                        else:
                            logger.warning("No research sources returned for query: %s", prompt)
                except (httpx.HTTPError, RuntimeError, ValueError):
                    research_results = []
                    research = ""
                    research_blocked = True
                    logger.warning("Research provider failed for query: %s", prompt)

        context = _build_context(prompt, history or [], knowledge=knowledge, research=research)
        language = understanding.language
        allow_romanized = _is_romanized_request(prompt, language)
        minimal_context = format_instruction_prompt(
            f"{_language_instruction(language)}\nUser request: {prompt}"
        )

        # Open-ended learning requests should still receive a useful answer when
        # the tiny local model is unavailable or produces unusable output.
        if _is_learning_request(prompt):
            return _learning_fallback_reply(language)

        # A strongly matched approved local fact can answer immediately without
        # paying the CPU cost of model loading/generation.
        knowledge_answer = _knowledge_fallback_sentence(prompt, knowledge_hits)
        if knowledge_answer is not None and not (research_blocked and intent.needs_research):
            return append_sources(knowledge_answer, _evidence_from_results(research_results))

        # Use a bounded public-knowledge lookup for ordinary factual questions.
        # This avoids spending the tiny Render CPU budget on a full model run
        # when a deterministic source can answer the question directly.
        if (
            not (research_blocked and intent.needs_research)
            and is_general_knowledge_question(prompt)
        ):
            try:
                web_answer = await _general_knowledge_provider.answer(
                    knowledge_query,
                    language=language,
                )
                if web_answer is not None:
                    return append_sources(
                        web_answer.extract,
                        [GroundedEvidence(web_answer.title, web_answer.url, web_answer.extract)],
                    )
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                logger.warning("General knowledge lookup failed: %s", exc)

        # Simple greetings should never pay the cost of loading/running the
        # trained local model. They are served immediately by the safe fallback.
        if _is_fast_fallback_message(prompt):
            return _fallback_reply(prompt)

        generation_failed = False
        try:
            # Model loading and CPU inference are blocking operations. Keep them
            # off FastAPI's event loop so a slow first load cannot stall the
            # service/proxy and turn an otherwise recoverable request into 502.
            runtime = await asyncio.wait_for(
                asyncio.to_thread(_load_local_model_runtime),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Indoone model load timed out; returning safe fallback")
            runtime = None
        except Exception:
            logger.exception("Indoone model preparation failed")
            runtime = None

        if runtime is None:
            try:
                answer = _clean_model_reply(await _fallback_engine.generate(context))
            except Exception:
                logger.exception("Indoone lightweight fallback engine failed")
                answer = _fallback_reply(prompt)
            knowledge_answer = _knowledge_fallback_sentence(knowledge_query, knowledge_hits)
            if knowledge_answer is not None:
                answer = knowledge_answer
            elif intent.needs_research and answer == _fallback_reply(prompt):
                evidence_answer = _research_extract_fallback(research_results)
                if evidence_answer is not None:
                    answer = evidence_answer
        else:
            answer = ""
            if intent.needs_research:
                # Fresh research must be synthesized from the gathered evidence.
                # Never try an evidence-free minimal prompt first.
                generation_contexts = [context]
            else:
                generation_contexts = [minimal_context]
                if context != minimal_context:
                    generation_contexts.append(context)
            temperatures, max_new_tokens = _generation_profile(prompt, intent.name)
            generation_deadline = time.monotonic() + _MODEL_GENERATION_TIMEOUT_SECONDS
            attempt = 0

            for generation_context in generation_contexts:
                for temperature in temperatures:
                    attempt += 1
                    remaining = generation_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = _clean_model_reply(
                            await asyncio.wait_for(
                                asyncio.to_thread(
                                    _generate_with_local_model,
                                    runtime,
                                    generation_context,
                                    language,
                                    temperature,
                                    max_new_tokens,
                                ),
                                timeout=remaining,
                            )
                        )
                        if not _has_expected_script(
                            candidate,
                            language,
                            allow_romanized=allow_romanized,
                        ):
                            logger.warning(
                                "Discarding malformed or wrong-language model output for %s on local generation attempt %d",
                                language,
                                attempt,
                            )
                            continue
                        quality = assess_answer(prompt, candidate)
                        if not quality.passed:
                            logger.warning(
                                "Discarding low-quality local model output on attempt %d: %s",
                                attempt,
                                quality.reason,
                            )
                            continue
                        answer = candidate
                        break
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Indoone local model generation timed out after %.1fs on attempt %d",
                            _MODEL_GENERATION_TIMEOUT_SECONDS,
                            attempt,
                        )
                        break
                    except Exception:
                        logger.exception(
                            "Indoone local model generation failed on attempt %d",
                            attempt,
                        )
                        continue
                if answer or generation_deadline - time.monotonic() <= 0:
                    break

            if not answer:
                answer = _generation_error_reply(language)
                knowledge_answer = _knowledge_fallback_sentence(knowledge_query, knowledge_hits)
                if knowledge_answer is not None:
                    answer = knowledge_answer
                else:
                    generation_failed = True

        # Fresh/current questions must not receive a normal model answer without
        # the required research evidence. The model may still run so the request
        # path remains testable, but its answer is discarded.
        if research_blocked and intent.needs_research:
            answer = user_safe_failure()
            research_results = []
        elif generation_failed and intent.needs_research:
            # Do not attach research links to a failed synthesis; sources must
            # support a user-visible answer rather than accompany an error message.
            research_results = []

        return append_sources(answer, _evidence_from_results(research_results))


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
) -> str:
    return await LocalAIService().generate(message, history=history, document_context=document_context)
