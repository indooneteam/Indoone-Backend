"""High-confidence local instruction retrieval for behavioral consistency."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.ai.training_data import TrainingExample, load_examples


_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "give", "how", "i", "if",
    "in", "is", "it", "me", "my", "no", "of", "on", "or", "please", "right",
    "say", "should", "so", "that", "the", "their", "them", "then", "there",
    "this", "to", "use", "what", "when", "which", "with", "would", "you",
    "your", "user", "assistant", "answer", "request", "tell", "about",
    "explain", "only", "just", "now", "very", "well", "while", "someone",
    "another", "through", "without", "into", "after", "earlier",
}


def _tokens(text: str) -> set[str]:
    return {
        _canonical_token(token.casefold())
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _canonical_token(token: str) -> str:
    aliases = {
        "actions": "action",
        "accounts": "account",
        "authorized": "authorize",
        "authorization": "authorize",
        "permissions": "permission",
        "services": "service",
        "sources": "source",
        "systems": "system",
        "tokens": "token",
        "disagreement": "disagree",
        "disagreements": "disagree",
        "disagrees": "disagree",
        "disagreed": "disagree",
        "conflicting": "conflict",
        "conflicts": "conflict",
        "researched": "research",
        "researching": "research",
        "summarize": "summary",
        "summarized": "summary",
        "summarization": "summary",
        "followup": "followup",
    }
    return aliases.get(token, token)


def _score(query: str, candidate: str) -> float:
    query_text = _normalized(query)
    candidate_text = _normalized(candidate)
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    if len(overlap) < 2:
        return 0.0

    recall = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(candidate_tokens)
    score = 0.65 * recall + 0.35 * precision

    if candidate_text == query_text:
        score += 0.35
    elif candidate_text in query_text or query_text in candidate_text:
        score += 0.20

    return min(score, 1.0)


@dataclass(frozen=True)
class RetrievedInstruction:
    example: TrainingExample
    score: float


class InstructionRetriever:
    """Retrieve a close curated instruction without model inference."""

    DEFAULT_SOURCES = (
        Path("data/raw/indoone_instructions.jsonl"),
        Path("data/raw/core_instruction_seed.jsonl"),
        Path("data/raw/indoone_phone_contacts_examples.jsonl"),
    )

    def __init__(self, sources: tuple[Path, ...] | None = None) -> None:
        self.sources = sources or self.DEFAULT_SOURCES
        examples: list[TrainingExample] = []
        seen: set[tuple[str, str]] = set()

        for source in self.sources:
            if not source.exists():
                continue
            for example in load_examples(source):
                key = (example.instruction.casefold(), example.response.casefold())
                if key in seen:
                    continue
                seen.add(key)
                examples.append(example)

        self.examples = tuple(examples)

    @property
    def available(self) -> bool:
        return bool(self.examples)

    @staticmethod
    def _category_hint(prompt: str) -> str | None:
        normalized = _normalized(prompt)
        research_markers = (
            "research",
            "source",
            "evidence",
            "current",
            "recent",
            "verified",
            "follow-up",
            "followup",
        )
        if any(marker in normalized for marker in research_markers):
            return "research"

        conversation_markers = (
            "concise",
            "remember a preference",
            "previous turn",
            "earlier context",
            "conversation",
        )
        if any(marker in normalized for marker in conversation_markers):
            return "conversation"
        return None

    def retrieve_many(
        self,
        prompt: str,
        *,
        minimum_score: float = 0.30,
        limit: int = 3,
    ) -> list[RetrievedInstruction]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        category_hint = self._category_hint(prompt)
        ranked: list[RetrievedInstruction] = []
        for example in self.examples:
            candidate_score = _score(prompt, example.instruction)
            if category_hint and example.category == category_hint:
                candidate_score += 0.18
            if candidate_score >= minimum_score:
                ranked.append(
                    RetrievedInstruction(example, min(candidate_score, 1.0))
                )

        ranked.sort(
            key=lambda item: (-item.score, item.example.instruction.casefold())
        )
        return ranked[:limit]

    def retrieve(self, prompt: str, *, minimum_score: float = 0.42) -> RetrievedInstruction | None:
        matches = self.retrieve_many(
            prompt,
            minimum_score=minimum_score,
            limit=1,
        )
        return matches[0] if matches else None
