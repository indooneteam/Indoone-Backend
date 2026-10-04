from __future__ import annotations

from app.ai.service import generate_reply


_IDENTITY_MARKERS = (
    "who are you",
    "what are you",
    "who made you",
    "who created you",
    "who created indoone",
    "who developed you",
    "who developed indoone",
    "who built you",
    "who built indoone",
    "who owns you",
    "who is your developer",
    "who is your creator",
    "which company made you",
    "which company developed you",
    "which company created you",
    "what company is behind you",
    "are you google",
    "are you gemini",
    "is google your developer",
    "is gemini your developer",
    "what ai model are you",
    "which ai model are you",
    "which model are you",
    "ನೀನು ಯಾರು",
    "ನೀವು ಯಾರು",
    "ನಿಮ್ಮನ್ನು ಯಾರು ತಯಾರಿಸಿದ್ದಾರೆ",
    "ನಿಮ್ಮನ್ನು ಯಾರು ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ್ದಾರೆ",
    "ನಿಮ್ಮನ್ನು ಯಾರು ಮಾಡಿದರು",
    "ನಿಮ್ಮ ಡೆವಲಪರ್ ಯಾರು",
    "ninu yaaru",
    "neenu yaaru",
    "nivu yaaru",
    "nimage yaaru",
    "ninna developer yaaru",
    "nimmannu yaaru madidaru",
    "nimmannu yaaru develop madidaru",
    "yav company ninna madide",
    "yaaru ninna develop madidru",
)


def _is_identity_question(message: str) -> bool:
    normalized = " ".join(message.casefold().split())
    return any(marker in normalized for marker in _IDENTITY_MARKERS)


def _identity_reply(message: str) -> str:
    normalized = " ".join(message.casefold().split())
    if any("\u0c80" <= char <= "\u0cff" for char in message):
        return "ನಾನು Indoone AI. ನಾನು Indoone ತಂಡ ಅಭಿವೃದ್ಧಿಪಡಿಸಿದ AI assistant."
    if any(
        marker in normalized
        for marker in ("ninu", "neenu", "nivu", "nimage", "ninna", "nimmannu")
    ):
        return "Naanu Indoone AI. Naanu Indoone team develop madida AI assistant."
    return "I’m Indoone AI, an AI assistant developed for Indoone."


async def generate_whatsapp_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    # WhatsApp has its own product identity policy. Keep this separate from the
    # shared AI service so other Indoone surfaces keep their existing behavior.
    if _is_identity_question(cleaned):
        return _identity_reply(cleaned)

    return await generate_reply(cleaned, history=history or [])
