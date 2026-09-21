from __future__ import annotations

from pathlib import Path

import torch

from app.ai.inference import LocalModelRuntime
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")
DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.0


class LocalAIEngine:
    """Load and run an Indoone-owned local language model."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        tokenizer_path: Path = DEFAULT_TOKENIZER,
    ) -> None:
        self.checkpoint = checkpoint
        self.tokenizer_path = tokenizer_path
        self.model: IndooneTransformer | None = None
        self.tokenizer: BPETokenizer | None = None
        self._runtime: LocalModelRuntime | None = None
        self._load_error: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.checkpoint.exists() or not self.tokenizer_path.exists():
            self._load_error = (
                "Indoone local model is not trained yet. Run "
                "python -m app.ai.train --corpus data/processed/train.txt "
                "--output models/indoone-small"
            )
            return

        try:
            self._runtime = LocalModelRuntime(self.checkpoint, self.tokenizer_path)
            self.model = self._runtime.model
            self.tokenizer = self._runtime.tokenizer
        except Exception as exc:
            self._runtime = None
            self._load_error = f"failed to load Indoone model: {exc}"

    @property
    def ready(self) -> bool:
        """Return whether the local checkpoint loaded successfully."""

        return (
            self.model is not None
            and self.tokenizer is not None
            and self._load_error is None
        )

    @staticmethod
    def _select_next_token(logits: torch.Tensor, temperature: float) -> int:
        """Select the next token, using greedy decoding at temperature 0."""

        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        if temperature == 0:
            return int(torch.argmax(logits, dim=-1).item())

        probabilities = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Encode a raw prompt using the shared inference prompt contract."""

        if self._runtime is not None:
            return self._runtime._prompt_ids(prompt)
        if self.tokenizer is None:
            raise RuntimeError(self._load_error or "Indoone local model is unavailable")
        stripped_prompt = prompt.strip()
        formatted_prompt = (
            stripped_prompt
            if stripped_prompt.startswith("<instruction>") and stripped_prompt.endswith("<response>")
            else (
                "<instruction>\n"
                f"{stripped_prompt}\n"
                "</instruction>\n"
                "<response>\n"
            )
        )
        token_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        return [self.tokenizer.stoi["<bos>"]] + token_ids

    @torch.inference_mode()
    async def generate(
        self,
        message: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate only the assistant completion for a prepared prompt."""

        if not self.ready:
            raise RuntimeError(
                self._load_error or "Indoone local model is unavailable"
            )

        if self._runtime is None:
            raise RuntimeError(self._load_error or "Indoone local model is unavailable")
        return self._runtime.generate(
            message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
