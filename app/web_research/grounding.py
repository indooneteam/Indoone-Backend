"""Grounding helpers for evidence-backed Indoone responses."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.ai.answer_quality import assess_answer


logger = logging.getLogger(__name__)


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
        "Synthesize all relevant evidence into one coherent answer; do not give "
        "separate source-by-source answers. Do not invent facts, sources, URLs, "
        "or details that are not supported by the conversation or supplied evidence. "
        "Use precise dates, quantities, percentages, and statistics only when they "
        "are supported by the supplied evidence; otherwise qualify the uncertainty "
        "or omit the precise value. "
        "When evidence is missing or conflicting, state the disagreement clearly "
        "instead of silently choosing one source.\n"
        "</grounding>"
    )


def assess_grounding(answer: str, evidence: list[GroundedEvidence]) -> GroundingQuality:
    """Check numeric anchors and source URLs against supplied evidence.

    Search snippets are incomplete, so an unmatched numeric anchor is a weak
    warning rather than definitive proof of a false claim. The evaluator can
    still flag it; production response handling must not turn that advisory into
    an HTTP 503. URLs in the answer can be checked deterministically and remain
    a hard validation failure when they were not returned by the search tool.
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


def extract_sources(answer: str) -> list[GroundedEvidence]:
    """Extract the deterministic source list for clients that render source cards."""
    cleaned = answer.strip()
    if not cleaned:
        return []

    parts = re.split(r"\n\s*sources:\s*\n", cleaned, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []

    sources: list[GroundedEvidence] = []
    for line in parts[1].splitlines():
        match = re.match(r"^\s*\d+\.\s+(.+?)\s+—\s+(https?://\S+)\s*$", line.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        title = match.group(1).strip()
        url = match.group(2).rstrip(".,;:")
        if title and url:
            sources.append(GroundedEvidence(title, url))
    return sources


def append_sources(answer: str, evidence: list[GroundedEvidence]) -> str:
    """Validate response quality, reject ungrounded URLs, and append source attribution."""

    cleaned = answer.strip()
    quality = assess_answer("", cleaned)
    if not quality.passed:
        raise RuntimeError(f"model answer failed quality checks: {quality.reason}")

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
        if grounding.reason == "unsupported_source_url":
            raise RuntimeError(f"model answer failed grounding checks: {grounding.reason}")
        # A lexical mismatch against short snippets is not proof the claim is false.
        # Keep the chat available, retain the verified source list, and log the
        # advisory; the prompt already asks the model to avoid unsupported precision.
        logger.warning("research grounding advisory: %s", grounding.reason)

    if not unique:
        return cleaned

    lines = [cleaned, "", "Sources:"]
    for index, item in enumerate(unique, start=1):
        title = item.title.strip() or item.url.strip()
        lines.append(f"{index}. {title} — {item.url.strip()}")
    return "\n".join(lines)


def split_grounded_sources(answer: str) -> tuple[str, list[GroundedEvidence]]:
    """Split a validated numbered Sources footer for clients that render source cards."""
    cleaned = answer.strip()
    parts = re.split(r"\n\s*sources:\s*\n", cleaned, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return answer, []
    sources = extract_sources(cleaned)
    if not sources:
        return answer, []
    return parts[0].rstrip(), sources
