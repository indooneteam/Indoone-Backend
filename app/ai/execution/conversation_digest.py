"""A digest of the thread, for when the user asks what was discussed.

The model sees four messages per turn, so a question about earlier in the conversation
cannot be answered from its payload. Sending the whole thread to answer it would undo the
saving on every turn that asks.

So: the question is recognised by its words, the stored thread is read, and a short digest
is returned - topics and decisions, not a transcript. No model call, no tokens.
"""

from __future__ import annotations

import re

from app.ai.execution.text_match import is_pleasantry
from app.ai.execution.intent_normalizer import normalize
from app.ai.memory.conversation_store import ConversationStore

# Asking about the conversation itself, rather than about the world.
_TRIGGERS = re.compile(
    r"(conversation\s*(check|summary|history)|chat\s*history|"
    r"what\s+did\s+we\s+discuss|discuss(ed)?\s+earlier|earlier\s+(chat|conversation)|"
    r"namma\s+(maatu|conversation)|hindina\s+maatu|"
    r"ಸಂಭಾಷಣೆ|ಹಿಂದಿನ\s*ಮಾತು|ಚಾಟ್\s*ಹಿಸ್ಟರಿ)",
    re.IGNORECASE,
)

# Words that mark an answer as having settled something.
_DECISION = re.compile(
    r"\b(confirmed|created|merged|deployed|sent|deleted|cancelled|canceled|fixed|"
    r"changed|added|removed|done)\b|ಮಾಡಲಾಗಿದೆ|ಕಳುಹಿಸಲಾಗಿದೆ|ರದ್ದು",
    re.IGNORECASE,
)

_TOPIC_LIMIT = 5
_DECISION_LIMIT = 3
_DIGEST_WINDOW = 15


def _trim(text: str, limit: int) -> str:
    cleaned = " ".join(str(text or "").split())

    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def digest(messages: list[tuple[str, str]], *, kannada: bool = False) -> str:
    """Topics and decisions from a stored thread, or a plain statement that there is none."""
    if not messages:
        return (
            "ಈ ಸಂಭಾಷಣೆಯಲ್ಲಿ ಇನ್ನೂ ಏನೂ ಉಳಿಸಲಾಗಿಲ್ಲ."
            if kannada
            else "Nothing is stored for this conversation yet."
        )

    window = messages[-_DIGEST_WINDOW:]
    topics: list[str] = []

    for role, text in window:
        if role != "user":
            continue

        topic = _trim(text, 90)

        if topic and topic not in topics:
            topics.append(topic)

    decisions = [
        _trim(text, 120)
        for role, text in window
        if role == "assistant" and _DECISION.search(str(text or ""))
    ]

    if kannada:
        lines = [f"ಇತ್ತೀಚಿನ ಸಂಭಾಷಣೆ ({len(window)} ಸಂದೇಶಗಳು):", "", "ವಿಷಯಗಳು:"]
        lines += [f"- {topic}" for topic in topics[-_TOPIC_LIMIT:]] or ["- ಇಲ್ಲ"]
        lines += ["", "ನಿರ್ಧಾರಗಳು:"]
        lines += [f"- {item}" for item in decisions[-_DECISION_LIMIT:]] or ["- ಯಾವುದೇ ನಿರ್ಧಾರ ದಾಖಲಾಗಿಲ್ಲ"]
        return "\n".join(lines)

    lines = [f"Recent conversation ({len(window)} messages):", "", "Topics:"]
    lines += [f"- {topic}" for topic in topics[-_TOPIC_LIMIT:]] or ["- none"]
    lines += ["", "Decisions:"]
    lines += [f"- {item}" for item in decisions[-_DECISION_LIMIT:]] or ["- none recorded"]
    return "\n".join(lines)


async def conversation_history_answer(
    user_id: str, conversation_id: str, message: str
) -> str | None:
    """The digest for an explicit question about the thread, or None to let it through."""
    text = (message or "").strip()

    if not text:
        return None

    # A greeting is not a question about the thread.
    if is_pleasantry(text):
        return None

    # "namma maatu helu", "conversation thorisu": the noun is here and the verb is a read,
    # which is the same request as "what did we discuss" in English.
    _norm = normalize(text)

    if not _TRIGGERS.search(text) and not (_norm.is_history and _norm.is_read):
        return None

    if not user_id.strip() or not conversation_id.strip():
        return None

    try:
        messages = ConversationStore().recent(conversation_id, user_id)
    except (ValueError, PermissionError):
        return None

    kannada = bool(re.search(r"[\u0C80-\u0CFF]", text))

    return digest(messages, kannada=kannada)
