from __future__ import annotations

import json
from pathlib import Path


class CharacterTokenizer:
    """Deterministic character-level tokenizer owned by Indoone."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        chars = sorted(set(text))
        if not chars:
            raise ValueError("training corpus is empty")
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            raise ValueError(f"text contains unknown characters: {unknown[:3]}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "CharacterTokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(list(payload["chars"]))
