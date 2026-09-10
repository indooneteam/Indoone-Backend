"""Grounding helpers for evidence-backed Indoone responses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundedEvidence:
    """Evidence supplied to the model and the user-facing source list."""

    title: str
    url: str
    snippet: str = ""


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


def append_sources(answer: str, evidence: list[GroundedEvidence]) -> str:
    """Append deterministic source attribution to a research-backed answer."""

    cleaned = answer.strip()
    if not evidence:
        return cleaned
    unique: list[GroundedEvidence] = []
    seen: set[str] = set()
    for item in evidence:
        key = item.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
    if not unique:
        return cleaned

    lines = [cleaned, "", "Sources:"]
    for index, item in enumerate(unique, start=1):
        title = item.title.strip() or item.url.strip()
        lines.append(f"{index}. {title} — {item.url.strip()}")
    return "\n".join(lines)
