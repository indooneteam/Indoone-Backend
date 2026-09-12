from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    needs_research: bool = False
    needs_file_context: bool = False
    needs_calculation: bool = False
    needs_cross_check: bool = False


_FRESH_TERMS = (
    "latest",
    "today",
    "current",
    "currently",
    "recent",
    "news",
    "right now",
    "this week",
    "this month",
    "this year",
    "now",
    "search",
    "look up",
    "research",
    "price",
    "cost",
    "stock",
    "weather",
    "forecast",
    "score",
    "schedule",
    "availability",
    "exchange rate",
    "currency rate",
    "market",
    "election",
    "president",
    "prime minister",
    "minister",
    "law",
    "regulation",
    "policy",
    "deadline",
    "release date",
    "version",
    "update",
    "ಯಾವಾಗ",
    "ಇವತ್ತು",
    "ಇತ್ತೀಚಿನ",
    "ಈಗ",
    "ದರ",
    "ಹವಾಮಾನ",
    "ಫಲಿತಾಂಶ",
    "ಬೆಲೆ",
)

_STABLE_LOOKUP_EXCEPTIONS = (
    "what is",
    "who is",
    "explain",
    "define",
    "meaning",
    "how does",
    "why does",
)


def should_use_fresh_research(message: str) -> bool:
    text = " ".join(message.strip().lower().split())
    if not text:
        return False
    return any(term in text for term in _FRESH_TERMS)


def should_cross_check(message: str) -> bool:
    text = " ".join(message.strip().lower().split())
    if not text:
        return False
    high_risk_terms = (
        "price", "cost", "stock", "weather", "forecast", "score", "election",
        "law", "regulation", "policy", "exchange rate", "currency rate",
        "availability", "schedule", "deadline", "current", "latest", "news",
    )
    return any(term in text for term in high_risk_terms)


def classify_intent(message: str) -> Intent:
    text = message.strip().lower()
    needs_research = should_use_fresh_research(text)
    needs_cross_check = should_cross_check(text)

    calc_words = ("calculate", "what is", "=", "+", "-", "*", "/", "ಗಣನೆ", "ಲೆಕ್ಕ")
    file_words = ("pdf", "document", "file", "attached", "ಡಾಕ್ಯುಮೆಂಟ್", "ಫೈಲ್")
    if needs_research:
        return Intent("research", needs_research=True, needs_cross_check=needs_cross_check)
    if any(word in text for word in file_words):
        return Intent("file_qa", needs_file_context=True)
    if any(word in text for word in calc_words) and any(ch.isdigit() for ch in text):
        return Intent("calculation", needs_calculation=True)
    if text.startswith(("translate ", "translate:", "ಅನುವಾದ")):
        return Intent("translation")
    if text.startswith(("summarize", "summary", "ಸಾರಾಂಶ")):
        return Intent("summarization")
    if text.startswith(("code", "debug", "fix this", "ಕೋಡ್")):
        return Intent("coding")
    return Intent("general")
