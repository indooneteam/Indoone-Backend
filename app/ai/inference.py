from __future__ import annotations

from pathlib import Path

import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.0


class LocalModelRuntime:
    """Load an Indoone checkpoint and generate text locally."""

    def __init__(self, checkpoint_path: Path, tokenizer_path: Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        config = dict(checkpoint["config"])
        config.pop("model_version", None)

        self.model = IndooneTransformer(
            vocab_size=self.tokenizer.vocab_size,
            **config,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

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
        """Encode a prompt with BOS but without a trailing EOS token.

        EOS belongs at the end of generated text. Putting EOS in the input
        prompt makes the model believe the conversation has already ended.
        """

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [self.tokenizer.stoi["<bos>"]] + token_ids

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate only the assistant completion for a prepared prompt."""

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
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
