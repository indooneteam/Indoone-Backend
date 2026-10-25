from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class BPETokenizer:
    """Small provider-independent BPE tokenizer with a fast Rust backend."""

    SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")

    def __init__(
        self,
        vocab: list[str] | None = None,
        merges: list[tuple[str, str]] | None = None,
        *,
        backend_tokenizer: Any | None = None,
    ) -> None:
        self._backend_tokenizer = backend_tokenizer
        if backend_tokenizer is not None:
            vocab_map = backend_tokenizer.get_vocab()
            self.vocab = [
                token
                for token, _ in sorted(vocab_map.items(), key=lambda item: item[1])
            ]
            self.stoi = dict(vocab_map)
            self.itos = {index: token for token, index in vocab_map.items()}
            self.merges = []
            self.unk_token = "<unk>"
            return

        if not vocab:
            raise ValueError("tokenizer vocabulary cannot be empty")
        self.vocab = list(vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        self.merges = list(merges or ())
        self.unk_token = "<unk>"

    @property
    def vocab_size(self) -> int:
        if self._backend_tokenizer is not None:
            return self._backend_tokenizer.get_vocab_size()
        return len(self.vocab)

    @staticmethod
    def _fast_backend() -> tuple[Any, Any, Any, Any, Any]:
        try:
            from tokenizers import Tokenizer
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
            from tokenizers.models import BPE
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.trainers import BpeTrainer
        except ImportError as exc:
            raise RuntimeError(
                "The 'tokenizers' package is required for BPE training. "
                "Install project requirements before training."
            ) from exc
        return Tokenizer, BPE, ByteLevel, ByteLevelDecoder, BpeTrainer

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

        Tokenizer, BPE, ByteLevel, ByteLevelDecoder, BpeTrainer = cls._fast_backend()
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(cls.SPECIAL_TOKENS),
            show_progress=True,
        )
        tokenizer.train_from_iterator([text], trainer=trainer)
        return cls(backend_tokenizer=tokenizer)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self._backend_tokenizer is None:
            base = [ch if ch in self.stoi else self.unk_token for ch in text]
            ids = self._apply_merges(base)
        else:
            ids = list(self._backend_tokenizer.encode(text).ids)

        if add_special_tokens:
            ids = [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        id_list = list(ids)
        if self._backend_tokenizer is not None:
            return self._backend_tokenizer.decode(
                id_list,
                skip_special_tokens=True,
            )
        special = set(self.SPECIAL_TOKENS)
        return "".join(
            self.itos[i] for i in id_list if self.itos[i] not in special
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._backend_tokenizer is not None:
            self._backend_tokenizer.save(str(path))
            return
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

        if payload.get("type") == "bpe":
            return cls(
                vocab=list(payload["vocab"]),
                merges=[tuple(pair) for pair in payload["merges"]],
            )

        model = payload.get("model", {})
        if model.get("type") != "BPE":
            raise ValueError("unsupported tokenizer format")

        Tokenizer, *_ = cls._fast_backend()
        tokenizer = Tokenizer.from_file(str(path))
        return cls(backend_tokenizer=tokenizer)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        for pair in self.merges:
            tokens = self._merge_pair(tokens, pair)
        return tokens

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
