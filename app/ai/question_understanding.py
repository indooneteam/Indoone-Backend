"""Lightweight, deterministic question-understanding layer for Indoone.

This layer intentionally avoids model inference and training. It turns common
natural-language variations into a small, stable decision object that the
runtime can use for routing, research, calculation, files, and response style.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QuestionUnderstanding:
    original: str
    normalized: str
    language: str
    intent: str
    needs_research: bool = False
    needs_cross_check: bool = False
    needs_file_context: bool = False
    needs_calculation: bool = False
    question_type: str = "general"
    research_query: str = ""


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

_ROMANIZED_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("bagge", "helu", "heLi", "maadu", "maadi", "madbeku", "enidu", "enu", "yenu", "ivaga", "matte", "nanage", "nimage", "nanna", "namma", "ide", "illa", "agide", "beku")),
    ("Hindi", ("kya", "hai", "hain", "mujhe", "aap", "aapko", "batao", "bataiye", "kaise", "kahan", "kyun", "nahi", "abhi", "mera", "meri")),
    ("Telugu", ("enti", "ela", "cheppu", "cheppandi", "undi", "ledu", "nenu", "meeru", "naaku", "emiti", "enduku", "ippudu", "malli")),
    ("Tamil", ("enna", "epdi", "eppadi", "sollu", "sollunga", "irukku", "illa", "naan", "neenga", "enakku", "ippo", "yen")),
    ("Malayalam", ("entha", "engane", "parayu", "parayoo", "undu", "illa", "njan", "ningal", "enikku", "ippo")),
    ("Marathi", ("kay", "aahe", "mala", "tumhi", "sanga", "kasa", "kashi", "kuthe", "nahi", "aata")),
    ("Gujarati", ("shu", "che", "chhe", "mane", "tame", "kaho", "kem", "nathi", "maru")),
    ("Punjabi", ("menu", "mainu", "tusi", "daso", "kive", "kiwe", "nahi", "mera", "sanu")),
)


_FRESH_MARKERS = (
    "latest", "today", "current", "currently", "recent", "news", "right now",
    "this week", "this month", "this year", "now", "search", "look up",
    "research", "stock", "weather", "forecast", "score", "schedule",
    "availability", "exchange rate", "currency rate", "market", "election",
    "president", "prime minister", "minister", "law", "regulation", "policy",
    "deadline", "release date", "version", "update",

)

_KANNADA_FRESH_MARKERS = (
    "ಇವತ್ತು", "ಇತ್ತೀಚಿನ", "ಈಗ", "ಈಗಿನ", "ಪ್ರಸ್ತುತ", "ಸದ್ಯ", "ಇಂದಿನ",
    "ರಾಷ್ಟ್ರಪತಿ", "ಪ್ರಧಾನಮಂತ್ರಿ", "ಮಂತ್ರಿ",
    "ಯಾಕೆ", "ಏನಕ್ಕೆ",
    "ದರ", "ಹವಾಮಾನ", "ಫಲಿತಾಂಶ", "ಬೆಲೆ",
)


_CROSS_CHECK_MARKERS = (
    "price", "cost", "stock", "weather", "forecast", "score", "election",
    "law", "regulation", "policy", "exchange rate", "currency rate",
    "availability", "schedule", "deadline", "current", "latest", "news",
)

_FILE_MARKERS = (
    "pdf", "document", "file", "attached", "attachment",
    "ಡಾಕ್ಯುಮೆಂಟ್", "ಫೈಲ್", "ಲಗತ್ತು",
)

_TRANSLATION_PREFIXES = ("translate ", "translate:", "ಅನುವಾದ", "अनुवाद", "మేరకు అనువద")
_SUMMARY_PREFIXES = ("summarize", "summary", "ಸಾರಾಂಶ", "सारांश")
_CODING_PREFIXES = ("code", "debug", "fix this", "ಕೋಡ್", "ಡಿಬಗ್")

_QUESTION_PREFIXES: tuple[tuple[str, str], ...] = (
    ("what is ", "definition"),
    ("what are ", "definition"),
    ("who is ", "who"),
    ("who was ", "who"),
    ("where is ", "where"),
    ("where was ", "where"),
    ("when is ", "when"),
    ("when was ", "when"),
    ("when did ", "when"),
    ("why is ", "why"),
    ("why are ", "why"),
    ("why does ", "why"),
    ("why do ", "why"),
    ("how does ", "how"),
    ("how do ", "how"),
    ("how is ", "how"),
    ("how many ", "how_many"),
    ("how much ", "how_much"),
    ("what causes ", "cause"),
    ("explain ", "explanation"),
    ("define ", "definition"),
    ("meaning of ", "definition"),
    ("difference between ", "comparison"),
    ("compare ", "comparison"),
    ("tell me about ", "about"),
    ("list ", "list"),
    ("andre enu", "definition"),
    ("enu", "definition"),
    ("yaaru", "who"),
    ("yaake", "why"),
    ("hege", "how"),
)

_QUESTION_PREFIX_RE = re.compile(
    r"^(?:what\s+(?:is|are)|who\s+(?:is|was)|where\s+(?:is|was)|"
    r"when\s+(?:is|was|did)|why\s+(?:is|are|does|do)|"
    r"how\s+(?:does|do|is|many|much)|what\s+causes|explain|define|"
    r"meaning\s+of|difference\s+between|compare|tell\s+me\s+about|list)\s+",
    flags=re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "please",
    "research", "simple", "source", "sources", "summary", "tell", "the", "this",
    "to", "today", "what", "when", "with", "explain", "latest", "current",
    "currently", "recent", "information", "find", "search",
}


def normalize_question(message: str) -> str:
    """Normalize whitespace and surrounding punctuation without changing meaning."""
    return " ".join(message.strip().split()).strip("?!.")


def detect_language(message: str) -> str:
    """Detect the user's likely response language without loading a model."""
    normalized = " ".join(message.casefold().split())
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


def _question_type(normalized: str) -> str:
    lower = normalized.casefold()
    for prefix, question_type in _QUESTION_PREFIXES:
        if lower.startswith(prefix):
            return question_type
    return "general"


def _topic_for_search(normalized: str) -> str:
    topic = _QUESTION_PREFIX_RE.sub("", normalized, count=1).strip(" ?!.")
    # Native Kannada question words can appear in the middle or at the end.
    for marker in ("ಎಂದರೇನು", "ಅರ್ಥವೇನು", "ಅರ್ಥ ಏನು", "ಎಷ್ಟು", "ಎಷ್ಟಿದೆ", "ಯಾವುದು", "ಯಾವಾಗ", "ಯಾವ", "ಯಾರು", "ಎಲ್ಲಿ", "ಏಕೆ", "ಯಾಕೆ", "ಏನಕ್ಕೆ", "ಹೇಗೆ", "ಏನು", "ಬಗ್ಗೆ", "ವಿವರಿಸಿ"):
        topic = re.sub(rf"(?<!\\S){re.escape(marker)}(?!\\S)", " ", topic)
    topic = " ".join(topic.split()).strip(" ?!.")
    romanized_tails = (" andre enu", " andre yen u", " enu", " yenu", " yaake", " hege", " yaaru")
    lower = topic.casefold()
    for tail in romanized_tails:
        if lower.endswith(tail) and len(topic) > len(tail):
            topic = topic[: -len(tail)].rstrip(" ?!.")
            break
    trailing = (
        " in simple words", " in simple terms", " in simple language",
        " briefly", " in short",
    )
    lower = topic.casefold()
    for phrase in trailing:
        if lower.endswith(phrase):
            topic = topic[: -len(phrase)].rstrip(" ?!.")
            break
    return topic or normalized


def _latin_topic_terms(text: str) -> str:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]*", text)
    filtered = [term for term in terms if term.casefold() not in _STOPWORDS]
    return " ".join(filtered).strip()


def _needs_research(normalized: str) -> bool:
    lower = normalized.casefold()
    historical_markers = (
        "first",
        "former",
        "formerly",
        "historical",
        "history",
        "was",
        "were",
        "ಹಿಂದಿನ",
        "ಮೊದಲ",
        "ಇತಿಹಾಸ",
        "ಭೂತಪೂರ್ವ",
    )
    if any(re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", lower) for marker in historical_markers):
        current_role_markers = ("president", "prime minister", "minister", "ರಾಷ್ಟ್ರಪತಿ", "ಪ್ರಧಾನಮಂತ್ರಿ", "ಮಂತ್ರಿ")
        if any(marker in lower for marker in current_role_markers):
            # Historical wording takes precedence over role-name freshness.
            return False

    for marker in _KANNADA_FRESH_MARKERS:
        if re.search(
            rf"(?<![\w\u0C80-\u0CFF]){re.escape(marker.casefold())}(?![\w\u0C80-\u0CFF])",
            lower,
        ):
            return True

    for marker in _FRESH_MARKERS:
        normalized_marker = marker.casefold()
        if normalized_marker == "ದರ":
            # Kannada vowel signs/combining marks are not all matched by Python \\w.
            if re.search(
                r"(?<![\w\u0C80-\u0CFF])ದರ(?![\w\u0C80-\u0CFF])",
                lower,
            ):
                return True
            continue
        if re.search(
            rf"(?<!\w){re.escape(normalized_marker)}(?!\w)",
            lower,
        ):
            return True
    return False
def _needs_cross_check(normalized: str) -> bool:
    lower = normalized.casefold()
    return any(marker in lower for marker in _CROSS_CHECK_MARKERS)


def _needs_calculation(normalized: str) -> bool:
    return bool(
        re.search(r"(?<!\w)(?:\d+(?:\.\d+)?\s*[+\-*/%]\s*)+\d+(?:\.\d+)?(?!\w)", normalized)
        and re.search(r"\d", normalized)
    ) or normalized.casefold().startswith(("calculate ", "ಗಣನೆ", "ಲೆಕ್ಕ"))


def understand_question(message: str) -> QuestionUnderstanding:
    """Return a routing decision using lightweight deterministic rules only."""
    normalized = normalize_question(message)
    if not normalized:
        return QuestionUnderstanding("", "", "English", "general")

    lower = normalized.casefold()
    language = detect_language(normalized)
    needs_research = _needs_research(normalized)
    needs_cross_check = _needs_cross_check(normalized)
    needs_file_context = any(marker in lower for marker in _FILE_MARKERS)
    needs_calculation = _needs_calculation(normalized)

    if needs_research:
        intent = "research"
    elif needs_file_context:
        intent = "file_qa"
    elif needs_calculation:
        intent = "calculation"
    elif any(lower.startswith(prefix.casefold()) for prefix in _TRANSLATION_PREFIXES):
        intent = "translation"
    elif any(lower.startswith(prefix.casefold()) for prefix in _SUMMARY_PREFIXES):
        intent = "summarization"
    elif any(lower.startswith(prefix.casefold()) for prefix in _CODING_PREFIXES):
        intent = "coding"
    else:
        intent = "general"

    topic = _topic_for_search(normalized)
    latin_topic = _latin_topic_terms(topic)
    research_query = latin_topic or topic

    return QuestionUnderstanding(
        original=message,
        normalized=normalized,
        language=language,
        intent=intent,
        needs_research=needs_research,
        needs_cross_check=needs_cross_check,
        needs_file_context=needs_file_context,
        needs_calculation=needs_calculation,
        question_type=_question_type(normalized),
        research_query=research_query,
    )
