from __future__ import annotations

from dataclasses import dataclass

from app.ai.question_understanding import understand_question


@dataclass(frozen=True)
class Intent:
    name: str
    needs_research: bool = False
    needs_file_context: bool = False
    needs_calculation: bool = False
    needs_cross_check: bool = False


def should_use_fresh_research(message: str) -> bool:
    """Return the deterministic routing decision for fresh-information requests."""
    return understand_question(message).needs_research


def should_cross_check(message: str) -> bool:
    """Return whether the request should use stronger source cross-checking."""
    return understand_question(message).needs_cross_check


def classify_intent(message: str) -> Intent:
    """Classify a request without loading or running the trained model."""
    understanding = understand_question(message)
    return Intent(
        name=understanding.intent,
        needs_research=understanding.needs_research,
        needs_file_context=understanding.needs_file_context,
        needs_calculation=understanding.needs_calculation,
        needs_cross_check=understanding.needs_cross_check,
    )
