from __future__ import annotations

from pathlib import Path
import re

import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = 1.08
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


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

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float,
    ) -> torch.Tensor:
        if penalty <= 1.0 or not generated_ids:
            return logits
        adjusted = logits.clone()
        for token_id in set(generated_ids[-128:]):
            value = adjusted[0, token_id]
            adjusted[0, token_id] = value * penalty if value < 0 else value / penalty
        return adjusted

    @staticmethod
    def _block_repeated_ngram(
        logits: torch.Tensor,
        generated_ids: list[int],
        ngram_size: int,
    ) -> torch.Tensor:
        if ngram_size < 2 or len(generated_ids) < ngram_size - 1:
            return logits
        prefix = tuple(generated_ids[-(ngram_size - 1):])
        blocked: set[int] = set()
        for index in range(len(generated_ids) - ngram_size + 1):
            ngram = tuple(generated_ids[index : index + ngram_size])
            if ngram[:-1] == prefix:
                blocked.add(ngram[-1])
        if blocked:
            logits = logits.clone()
            logits[0, list(blocked)] = float("-inf")
        return logits

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Render a user request in the same instruction format used for training."""

        stripped_prompt = prompt.strip()
        if stripped_prompt.startswith("<instruction>") and stripped_prompt.endswith("<response>"):
            formatted_prompt = stripped_prompt
        else:
            formatted_prompt = (
                "<instruction>\n"
                f"{stripped_prompt}\n"
                "</instruction>\n"
                "<response>\n"
            )
        token_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        return [self.tokenizer.stoi["<bos>"]] + token_ids

    @staticmethod
    def _clean_completion(text: str) -> str:
        """Keep only the assistant response and strip leaked training markers."""

        cleaned = text.strip()
        marker_match = re.search(r"</?(?:instruction|response|conversation|grounding|response_language)[^>]*>", cleaned, flags=re.IGNORECASE)
        if marker_match:
            cleaned = cleaned[:marker_match.start()].strip()
        for marker in ("USER:", "\nUSER:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        if cleaned.startswith("<response>"):
            cleaned = cleaned[len("<response>") :].strip()
        return cleaned

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        """Generate only the assistant completion for a prepared prompt."""

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must not be negative")

        prompt_ids = self._prompt_ids(prompt)
        generated_ids = list(prompt_ids)
        eos_id = self.tokenizer.stoi["<eos>"]

        for _ in range(max_new_tokens):
            context_ids = generated_ids[-self.model.block_size :]
            context = torch.tensor([context_ids], dtype=torch.long)
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :]
            next_logits = self._apply_repetition_penalty(
                next_logits,
                generated_ids,
                repetition_penalty,
            )
            next_logits = self._block_repeated_ngram(
                next_logits,
                generated_ids,
                no_repeat_ngram_size,
            )
            next_id = self._select_next_token(next_logits, temperature)

            generated_ids.append(next_id)
            if next_id == eos_id:
                break

        completion_ids = generated_ids[len(prompt_ids) :]
        return self._clean_completion(self.tokenizer.decode(completion_ids))
