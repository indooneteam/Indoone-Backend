from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerQuality:
    """Deterministic safety/quality checks for model output."""

    passed: bool
    reason: str = ""


_LEAK_MARKERS = (
    "fallback mode",
    "trained local model",
    "local model is not available",
    "backend is reachable",
    "indoone backend is reachable",
    "<instruction>",
    "</instruction>",
    "<response>",
    "</response>",
)

_CURRENT_MARKERS = (
    "latest",
    "today",
    "current",
    "currently",
    "recent",
    "right now",
    "this week",
    "news",
)

_SENTENCE_RE = re.compile(r"(?<=[.!?।!?])\s+")


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def assess_answer(question: str, answer: str) -> AnswerQuality:
    """Reject output that is empty, leaks internals, or is obviously malformed.

    This gate does not pretend to prove factual correctness. It only prevents
    known bad user-facing output from being returned as a successful answer.
    """

    question_text = _normalized(question)
    answer_text = _normalized(answer)

    if not answer_text:
        return AnswerQuality(False, "empty_answer")

    if len(answer_text) < 3:
        return AnswerQuality(False, "answer_too_short")

    for marker in _LEAK_MARKERS:
        if marker in answer_text:
            return AnswerQuality(False, "internal_detail_leak")

    sentences = [part.strip() for part in _SENTENCE_RE.split(answer.strip()) if part.strip()]
    if len(sentences) >= 3:
        normalized_sentences = [_normalized(part) for part in sentences]
        repeated = sum(
            1
            for index, sentence in enumerate(normalized_sentences)
            if sentence and sentence in normalized_sentences[:index]
        )
        if repeated >= 2:
            return AnswerQuality(False, "repeated_output")

    if any(marker in question_text for marker in _CURRENT_MARKERS):
        # Freshness is only considered reliable when the response includes the
        # service's explicit source section.
        if "sources:" not in answer_text:
            return AnswerQuality(False, "freshness_not_supported")

    return AnswerQuality(True)


def user_safe_failure() -> str:
    """Return a neutral message when the generated answer fails the gate."""

    return "I’m sorry, I don’t have enough reliable information to give you a confident answer right now."
