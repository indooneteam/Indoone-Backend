"""Curated instruction data helpers for the Indoone local model.

The training format is intentionally simple and provider-independent so the
corpus can grow without coupling Indoone to a hosted model service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


MAX_TEXT_LENGTH = 8_000


@dataclass(frozen=True)
class TrainingExample:
    instruction: str
    response: str
    category: str = "general"

    def as_text(self) -> str:
        return (
            "<instruction>\n"
            f"{self.instruction.strip()}\n"
            "</instruction>\n"
            "<response>\n"
            f"{self.response.strip()}\n"
            "</response>"
        )


def _clean_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = " ".join(value.split()).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} is too long")
    return text


def load_examples(path: Path) -> list[TrainingExample]:
    """Load and validate JSONL instruction/response examples."""

    examples: list[TrainingExample] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"example must be an object on line {line_number}")
        instruction = _clean_text(payload.get("instruction"), "instruction")
        response = _clean_text(payload.get("response"), "response")
        category = _clean_text(payload.get("category", "general"), "category")
        key = (instruction.casefold(), response.casefold())
        if key in seen:
            continue
        seen.add(key)
        examples.append(TrainingExample(instruction, response, category))
    if not examples:
        raise ValueError("training example file is empty")
    return examples


def write_corpus(examples: list[TrainingExample], output: Path) -> Path:
    """Render validated examples as deterministic training text."""

    if not examples:
        raise ValueError("examples cannot be empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n\n".join(example.as_text() for example in examples) + "\n", encoding="utf-8")
    return output
