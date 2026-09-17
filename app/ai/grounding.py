"""Grounding helpers for evidence-backed Indoone responses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.answer_quality import assess_answer, user_safe_failure


@dataclass(frozen=True)
class GroundedEvidence:
    """Evidence supplied to the model and the user-facing source list."""

    title: str
    url: str
    snippet: str = ""


@dataclass(frozen=True)
class GroundingQuality:
    """Deterministic evidence-consistency checks for grounded answers."""

    passed: bool
    reason: str = ""


_FACT_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d+(?:[.,]\d+)?%?|\d{4})(?![A-Za-z0-9])")
_URL_RE = re.compile(r"https?://[^\s)]+", flags=re.IGNORECASE)


def build_grounded_prompt_instruction() -> str:
    """Return a compact instruction that prioritizes supplied evidence."""

    return (
        "<grounding>\n"
        "Use the supplied knowledge and research evidence when relevant. "
        "Do not invent facts, sources, URLs, or details that are not supported "
        "by the conversation or supplied evidence. When evidence is missing or "
        "conflicting, say so clearly.\n"
        "</grounding>"
    )


def assess_grounding(answer: str, evidence: list[GroundedEvidence]) -> GroundingQuality:
    """Check concrete anchors in grounded output against supplied evidence.

    This is intentionally narrower than semantic factual verification: it catches
    unsupported numbers/years/percentages and user-visible URLs, while allowing
    natural explanatory prose that cannot be reliably validated with lexical
    matching alone.
    """

    cleaned = answer.strip()
    if not cleaned or not evidence:
        return GroundingQuality(True)

    evidence_text = "\n".join(
        f"{item.title}\n{item.snippet}" for item in evidence if item.title.strip() or item.snippet.strip()
    )
    answer_body = re.split(r"\n\s*sources:\s*", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]

    evidence_facts = {item.casefold() for item in _FACT_RE.findall(evidence_text)}
    unsupported_facts = [
        item for item in _FACT_RE.findall(answer_body) if item.casefold() not in evidence_facts
    ]
    if unsupported_facts:
        return GroundingQuality(False, "unsupported_concrete_fact")

    evidence_urls = {item.url.strip().rstrip(".,;:") for item in evidence if item.url.strip()}
    for url in _URL_RE.findall(answer_body):
        normalized = url.rstrip(".,;:")
        if normalized not in evidence_urls:
            return GroundingQuality(False, "unsupported_source_url")

    return GroundingQuality(True)


def append_sources(answer: str, evidence: list[GroundedEvidence]) -> str:
    """Quality-gate user-facing output and append deterministic source attribution."""

    cleaned = answer.strip()
    quality = assess_answer("", cleaned)
    if not quality.passed:
        return user_safe_failure()

    unique: list[GroundedEvidence] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    grounding = assess_grounding(cleaned, unique)
    if not grounding.passed:
        return user_safe_failure()

    if not unique:
        return cleaned

    lines = [cleaned, "", "Sources:"]
    for index, item in enumerate(unique, start=1):
        title = item.title.strip() or item.url.strip()
        lines.append(f"{index}. {title} — {item.url.strip()}")
    return "\n".join(lines)
