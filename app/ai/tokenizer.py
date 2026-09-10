from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class BPETokenizer:
    """Small in-house byte-pair tokenizer for the Indoone training pipeline."""

    SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]) -> None:
        if not vocab:
            raise ValueError("tokenizer vocabulary cannot be empty")
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.merges = list(merges)
        self.unk_token = "<unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        if not text:
            raise ValueError("training corpus is empty")
        if vocab_size < len(cls.SPECIAL_TOKENS):
            raise ValueError("vocab_size is smaller than the required special tokens")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least one")

        symbols = sorted(set(text))
        if len(cls.SPECIAL_TOKENS) + len(symbols) > vocab_size:
            raise ValueError(
                "vocab_size is too small for the corpus character alphabet; "
                "increase vocab_size"
            )

        sequences = [list(text)]
        merges: list[tuple[str, str]] = []
        vocab = list(cls.SPECIAL_TOKENS) + symbols

        while len(vocab) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))

            candidates = [
                (count, pair) for pair, count in pair_counts.items() if count >= min_frequency
            ]
            if not candidates:
                break

            _, best_pair = max(candidates, key=lambda item: (item[0], item[1]))
            merged_token = best_pair[0] + best_pair[1]
            if merged_token in vocab:
                break

            merges.append(best_pair)
            vocab.append(merged_token)
            sequences = [cls._merge_pair(sequence, best_pair) for sequence in sequences]

        return cls(vocab=vocab, merges=merges)

    @staticmethod
    def _merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
        merged: list[str] = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                merged.append(tokens[index] + tokens[index + 1])
                index += 2
            else:
                merged.append(tokens[index])
                index += 1
        return merged

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

    def _tokenize(self, text: str) -> list[str]:
        base = [ch if ch in self.stoi else self.unk_token for ch in text]
        return self._apply_merges(base)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.stoi[token] for token in self._tokenize(text)]
        if add_special_tokens:
            ids = [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        special = set(self.SPECIAL_TOKENS)
        return "".join(self.itos[i] for i in ids if self.itos[i] not in special)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "type": "bpe",
                    "version": 1,
                    "vocab": self.vocab,
                    "merges": [list(pair) for pair in self.merges],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("type") != "bpe":
            raise ValueError("unsupported tokenizer format")
        merges = [tuple(pair) for pair in payload["merges"]]
        return cls(vocab=list(payload["vocab"]), merges=merges)
