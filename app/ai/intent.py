from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    needs_research: bool = False
    needs_file_context: bool = False
    needs_calculation: bool = False


def classify_intent(message: str) -> Intent:
    text = message.strip().lower()
    research_words = ("latest", "today", "current", "news", "recent", "search", "ಯಾವಾಗ", "ಇವತ್ತು", "ಇತ್ತೀಚಿನ")
    calc_words = ("calculate", "what is", "=", "+", "-", "*", "/", "ಗಣನೆ", "ಲೆಕ್ಕ")
    file_words = ("pdf", "document", "file", "attached", "ಡಾಕ್ಯುಮೆಂಟ್", "ಫೈಲ್")
    if any(word in text for word in research_words):
        return Intent("research", needs_research=True)
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
