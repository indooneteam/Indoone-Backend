"""The capability gate: a question ABOUT a tool is not a request FOR the tool.

"what is email?" contains the word email. "explain how a PR works" contains a PR. Both used to
reach the mailbox or the GitHub API, because the routers matched nouns. The user asked for an
explanation and got their inbox - and, worse, an external call was made for a question that
needed none.

So this sits in front of the two capability routers and answers one question: **is this person
asking us to DO something with their data, or asking what something IS?**

The rule is deliberately narrow. It can only ever refuse; it never lets through something the
router would not have matched. A gate that loosens is a gate that invents requests.

Three ways a message earns the capability:

* an imperative that only makes sense against the real thing - "list", "show", "send", "merge"
* a possessive that points at the person's own data - "my", "ನನ್ನ", "mera"
* a router that already found a concrete target - a number, a name, an ordinal

A definition question with none of those is conversation, and the model answers it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

GITHUB = "github"
MAIL = "mail"
CONVERSATION = "conversation"

# A question about the concept, not about the person's data.
_DEFINITION = re.compile(
    r"^\s*(?:what\s+(?:is|are|does)|what's|explain|define|describe|how\s+(?:does|do|is)|"
    r"why\s+(?:does|do|is)|tell\s+me\s+about|meaning\s+of|difference\s+between)\b"
    r"|ಎಂದರೇನು|ಏನು\s+ಎಂದರೆ|ವಿವರಿಸಿ|ಹೇಗೆ\s+ಕೆಲಸ\s+ಮಾಡುತ್ತದೆ"
    r"|क्या\s+है|कैसे\s+काम\s+करता|समझाओ",
    re.IGNORECASE,
)

# Words that ask for the real thing rather than its definition.
_IMPERATIVE = re.compile(
    r"\b(?:list|show|display|send|draft|reply|read|open|fetch|create|delete|merge|close|reopen|"
    r"push|pull|check|review|approve|search|find|count|how\s+many|mark)\b"
    r"|ತೋರಿಸ|ಕಳಿಸ|ಬರೆ|ಓದು|ಎಷ್ಟು|ಇವೆಯಾ|ಇದೆಯಾ"
    r"|दिखाओ|भेजो|बनाओ|पढ़ो|कितने|हैं\s+क्या",
    re.IGNORECASE,
)

# The person's own data, which a definition question never has.
_POSSESSIVE = re.compile(
    r"\b(?:my|our|mine)\b|ನನ್ನ|ನಮ್ಮ|मेरा|मेरी|मेरे|हमारा",
    re.IGNORECASE,
)

# A concrete target: a number, a repository-looking name, a branch-looking ref.
_CONCRETE = re.compile(r"\b\d+\b|\b[\w.-]+/[\w.-]+\b|\b[A-Za-z]+-[A-Za-z0-9._-]{2,}\b")


@dataclass(slots=True)
class CapabilityDecision:
    capability: str = CONVERSATION
    confidence: float = 0.0
    reason: str = ""

    @property
    def is_capability(self) -> bool:
        return self.capability != CONVERSATION


def enabled() -> bool:
    return os.getenv("INDOONE_CAPABILITY_GATE_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def decide(text: str, *, capability: str) -> CapabilityDecision:
    """Whether a message that *matched* a capability word is actually asking for it."""
    body = (text or "").strip()

    if not body:
        return CapabilityDecision(CONVERSATION, 0.0, "empty")

    if _IMPERATIVE.search(body) or _POSSESSIVE.search(body) or _CONCRETE.search(body):
        return CapabilityDecision(capability, 0.9, "the message asks for the real thing")

    if _DEFINITION.search(body):
        # "what is email?" - a question the model answers, with no external call.
        return CapabilityDecision(CONVERSATION, 0.1, "a question about the concept")

    # The router matched a noun and nothing argues against it: leave the router's decision
    # alone rather than inventing a refusal.
    return CapabilityDecision(capability, 0.6, "the router's own match stands")


def wants(text: str, capability: str) -> bool:
    """The one call the routers make. False only when this is clearly conversation."""
    if not enabled():
        return True

    return decide(text, capability=capability).is_capability
