"""Training-data fallback for Email capability responses.

The trained local model remains the primary generator. When an Email request contains
an actual email payload but model generation is unavailable or fails its quality gate,
this module uses the repository's validated Email examples as a conservative fallback.
It never sends mail or performs external actions.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ai.training.training_data import load_examples

_ROOT = Path(__file__).resolve().parents[2]
_SAFETY_SOURCES = (
    _ROOT / "data/raw/indoone_email_safety_examples.jsonl",
    _ROOT / "data/raw/indoone_email_safety_additional_generated.jsonl",
)
_ACTION_SOURCES = (
    _ROOT / "data/raw/indoone_email_actions_examples.jsonl",
    _ROOT / "data/raw/indoone_email_actions_additional_generated.jsonl",
)

_WORD_RE = re.compile(r"[A-Za-z0-9_\u0C80-\u0CFF\u0900-\u097F\u0B80-\u0BFF\u0D00-\u0D7F]+")


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def is_email_request(prompt: str) -> bool:
    normalized = _normalized(prompt)
    return bool(
        re.search(
            r"(?:\bemail\b|\be-mail\b|\binbox\b|\bmailbox\b|"
            r"\bphishing\b|\bspam\b|\bunsubscribe\b|"
            r"\bsubject\s*:|\bfrom\s*:|\bto\s*:)",
            normalized,
        )
    )


def has_email_payload(prompt: str) -> bool:
    normalized = _normalized(prompt)
    structured_headers = sum(
        bool(re.search(rf"(?:^|\\s){header}\\s*:", normalized))
        for header in ("email", "subject", "from", "to")
    )
    return (
        "email:" in normalized
        or "untrusted data" in normalized
        or ("subject:" in normalized and "from:" in normalized)
        or structured_headers >= 2
    )


def needs_email_payload(prompt: str) -> bool:
    normalized = _normalized(prompt)
    classification_markers = (
        "phishing",
        "spam",
        "legitimate",
        "is this email",
        "check this email",
        "analyze this email",
        "email safety",
    )
    return is_email_request(prompt) and not has_email_payload(prompt) and any(
        marker in normalized for marker in classification_markers
    )


def _load_training_examples() -> list[object]:
    examples: list[object] = []
    seen: set[tuple[str, str]] = set()
    for path in (*_SAFETY_SOURCES, *_ACTION_SOURCES):
        if not path.is_file():
            continue
        for example in load_examples(path):
            key = (example.instruction.casefold(), example.response.casefold())
            if key in seen:
                continue
            seen.add(key)
            examples.append(example)
    return examples


def _tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(text)
        if len(token) >= 3
    }


def _email_similarity(prompt: str, instruction: str) -> float:
    query = _tokens(prompt)
    candidate = _tokens(instruction)
    if not query or not candidate:
        return 0.0
    overlap = len(query & candidate)
    recall = overlap / len(query)
    precision = overlap / len(candidate)
    return 0.70 * recall + 0.30 * precision


def training_fallback(prompt: str) -> str | None:
    """Return the closest validated Email training response for a concrete payload."""
    if not has_email_payload(prompt):
        return None

    examples = _load_training_examples()
    if not examples:
        return None

    ranked = sorted(
        (
            (_email_similarity(prompt, str(example.instruction)), example)
            for example in examples
            if getattr(example, "category", "") in {"email_safety", "email_compose", "email_reply", "email_send"}
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    if not ranked:
        return None

    score, example = ranked[0]
    if score < 0.08:
        return None

    return str(example.response).strip() or None


def input_guidance() -> str:
    return "Paste the email content (including Subject and From) and I’ll analyze whether it looks legitimate, spam, or phishing and explain the reasons."
