"""Indoone AI core entry point.

The service keeps inference local and provider-independent. A trained Indoone
checkpoint is used when available; otherwise the small local fallback keeps
the chat API available.
"""

from pathlib import Path
import logging
import re
from urllib.parse import urlparse

import httpx

from app.ai.grounding import (
    GroundedEvidence,
    append_sources,
    build_grounded_prompt_instruction,
)
from app.ai.inference import LocalModelRuntime
from app.ai.intent import classify_intent
from app.ai.knowledge import LocalKnowledgeBase, format_hits
from app.ai.local_engine import LocalAIEngine
from app.ai.research import (
    ResearchProvider,
    ResearchResult,
    build_research_provider,
    format_results,
)
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


_RESPONSE_TAG_RE = re.compile(r"</?(?:instruction|response|conversation|grounding|response_language)>|<response_language>.*?</response_language>", flags=re.IGNORECASE | re.DOTALL)


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
        score = sum(1 for hint in hints if re.search(rf"(?<!\w){re.escape(hint.casefold())}(?!\w)", normalized))
        if score:
            scores[language] = score
    if scores:
        best_language, best_score = max(scores.items(), key=lambda item: item[1])
        tied = [language for language, score in scores.items() if score == best_score]
        if best_score >= 2 or len(tied) == 1:
            return best_language
    return "English"


def _language_instruction(language: str) -> str:
    return f"Respond only in {language}. Preserve the user's language and script. Do not switch languages unless the user explicitly requests it. Keep the answer natural, clear, and concise."


def _build_context(message: str, history: list[tuple[str, str]], knowledge: str = "", research: str = "") -> str:
    response_language = _detect_response_language(message)
    grounding_instruction = build_grounded_prompt_instruction().replace("<grounding>", "").replace("</grounding>", "").strip()
    prompt_parts = ["<instruction>", grounding_instruction, _language_instruction(response_language)]
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
    prompt_parts.extend(("</instruction>", "<response>"))
    return "\n".join(prompt_parts)


def _evidence_from_results(results: list[ResearchResult]) -> list[GroundedEvidence]:
    return [GroundedEvidence(title=result.title, url=result.url, snippet=result.snippet) for result in results]


def _research_has_enough_sources(results: list[ResearchResult], cross_check: bool) -> bool:
    unique_domains = {urlparse(result.url).netloc.casefold() for result in results if urlparse(result.url).netloc}
    return len(unique_domains) >= (2 if cross_check else 1)


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
        text = text[: response_end.start()]
    text = _RESPONSE_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _has_expected_script(text: str, language: str) -> bool:
    if not text:
        return False
    if language == "English":
        return bool(re.search(r"[A-Za-z]", text))
    pattern = dict(_SCRIPT_RANGES).get(language)
    return pattern is not None and bool(pattern.search(text))


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
        "Odia": "ଦୁଃଖିତ, ମୁଁ ଏହି ପ୍ରଶ୍ନର ସଠିକ ଉତ୍ତର ଏବେ ଦେଇପାରିଲି ନାହିଁ। ଦୟାକରି ପୁଣି ପଚାରନ୍ତୁ।",
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
        return "ନମସ୍କାର 👋 ମୁଁ Indoone AI। trained local model ଏଯାବତ୍ load ହୋଇନାହିଁ।"
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

        intent = classify_intent(prompt)
        knowledge = ""
        if _knowledge_base is not None:
            knowledge = format_hits(_knowledge_base.search(prompt, limit=3))
        if document_context.strip():
            knowledge = (knowledge + "\n\n" if knowledge else "") + "User-provided document:\n" + document_context.strip()[:100_000]

        research = ""
        research_results: list[ResearchResult] = []
        if _research_provider is not None and intent.needs_research:
            try:
                candidate_results = await _research_provider.search(prompt, limit=5)
                if _research_has_enough_sources(candidate_results, intent.needs_cross_check):
                    research_results = candidate_results
                    research = format_results(candidate_results)
                elif candidate_results:
                    logger.warning("Insufficient independent research sources for query: %s", prompt)
                    research = ""
            except (httpx.HTTPError, RuntimeError, ValueError):
                research_results = []
                research = ""

        context = _build_context(prompt, history or [], knowledge=knowledge, research=research)
        language = _detect_response_language(prompt)

        if _runtime is not None:
            try:
                answer = _clean_model_reply(_runtime.generate(context))
                if not _has_expected_script(answer, language):
                    logger.warning("Discarding malformed or wrong-language model output for %s", language)
                    answer = _generation_error_reply(language)
            except RuntimeError:
                answer = _fallback_reply(prompt)
        elif getattr(_fallback_engine, "ready", True):
            try:
                answer = _clean_model_reply(await _fallback_engine.generate(context))
                if not _has_expected_script(answer, language):
                    answer = _generation_error_reply(language)
            except RuntimeError:
                answer = _fallback_reply(prompt)
        else:
            answer = _fallback_reply(prompt)
        return append_sources(answer, _evidence_from_results(research_results))


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
) -> str:
    return await LocalAIService().generate(message, history=history, document_context=document_context)
