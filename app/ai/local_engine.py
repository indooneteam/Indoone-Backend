from __future__ import annotations

from pathlib import Path

import torch

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
            payload = torch.load(
                self.checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            self.tokenizer = BPETokenizer.load(self.tokenizer_path)

            raw_config = dict(payload["config"])
            raw_config.pop("model_version", None)

            self.model = IndooneTransformer(
                vocab_size=self.tokenizer.vocab_size,
                **raw_config,
            )
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
        except Exception as exc:
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
        """Encode a prompt with BOS and leave EOS for generated output."""

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
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

        assert self.model is not None
        assert self.tokenizer is not None

        prompt = message.strip()
        if not prompt:
            raise ValueError("message cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        prompt_ids = self._prompt_ids(prompt)
        generated_ids = list(prompt_ids)
        eos_id = self.tokenizer.stoi["<eos>"]

        for _ in range(max_new_tokens):
            context_ids = generated_ids[-self.model.block_size :]
            context = torch.tensor([context_ids], dtype=torch.long)
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :]
            next_id = self._select_next_token(next_logits, temperature)

            generated_ids.append(next_id)
            if next_id == eos_id:
                break

        completion_ids = generated_ids[len(prompt_ids) :]
        return self.tokenizer.decode(completion_ids).strip()
