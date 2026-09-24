"""High-confidence local instruction retrieval for behavioral consistency."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.ai.training_data import TrainingExample, load_examples


_TOKEN_RE = re.compile(r"[^\\W_]+", flags=re.UNICODE)
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
        token.casefold()
        for token in _TOKEN_RE.findall(text)
        if len(token) >= 3 and token.casefold() not in _STOPWORDS
    }


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _score(query: str, candidate: str) -> float:
    query_text = _normalized(query)
    candidate_text = _normalized(candidate)
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = query_tokens & candidate_tokens
    if len(overlap) < 3:
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

    def retrieve(self, prompt: str, *, minimum_score: float = 0.42) -> RetrievedInstruction | None:
        best: RetrievedInstruction | None = None
        for example in self.examples:
            candidate_score = _score(prompt, example.instruction)
            if best is None or candidate_score > best.score:
                best = RetrievedInstruction(example, candidate_score)

        if best is None or best.score < minimum_score:
            return None
        return best
