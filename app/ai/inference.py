from __future__ import annotations

from pathlib import Path
import re

import torch

from app.ai.instruction_retrieval import InstructionRetriever
from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.training_data import format_instruction_prompt


DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = 1.08
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3


class LocalModelRuntime:
    """Load an Indoone local language model with a high-confidence answer fallback."""

    def __init__(self, checkpoint_path: Path, tokenizer_path: Path) -> None:
        # Render's free instance has a fractional CPU allocation. Limit Torch
        # intra-op parallelism so the runtime does not oversubscribe its CPU slice.
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Torch may already be initialized by another component.
            pass

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except (TypeError, RuntimeError, ValueError):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

        self.tokenizer = BPETokenizer.load(tokenizer_path)

        config = dict(checkpoint["config"])
        config.pop("model_version", None)

        self.model = IndooneTransformer(
            vocab_size=self.tokenizer.vocab_size,
            **config,
        )
        self.model.load_state_dict(checkpoint["model_state"], assign=True)

        # Re-establish the tied embedding/output weights after assign-based loading.
        self.model.lm_head.weight = self.model.token_embedding.weight
        del checkpoint
        self.model.eval()

        project_root = Path(__file__).resolve().parents[2]
        sources = tuple(
            project_root / relative
            for relative in InstructionRetriever.DEFAULT_SOURCES
        )
        self._instruction_retriever = InstructionRetriever(sources)

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
        """Penalize tokens already generated in the current completion only."""
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
        """Block repeated n-grams from the assistant completion."""
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

    @staticmethod
    def _extract_user_request(prompt: str) -> str:
        """Recover the final user message from a prepared inference context."""
        marker = re.findall(
            r"(?:^|\n)user:\s*(.+?)(?=\n(?:user|assistant):|\n</instruction>|\Z)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if marker:
            return marker[-1].strip()
        return prompt.strip()

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Render a user request in the same instruction format used for training."""
        stripped_prompt = prompt.strip()
        if stripped_prompt.startswith("<instruction>") and stripped_prompt.endswith("<response>"):
            formatted_prompt = stripped_prompt
        else:
            formatted_prompt = format_instruction_prompt(stripped_prompt)
        token_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        return [self.tokenizer.stoi["<bos>"]] + token_ids

    @staticmethod
    def _clean_completion(text: str) -> str:
        """Keep only the assistant response and strip leaked training markers."""
        cleaned = text.strip()
        marker_match = re.search(
            r"</?(?:instruction|response|conversation|grounding|response_language)[^>]*>",
            cleaned,
            flags=re.IGNORECASE,
        )
        if marker_match:
            cleaned = cleaned[:marker_match.start()].strip()
        for marker in ("USER:", "\nUSER:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        if cleaned.startswith("<response>"):
            cleaned = cleaned[len("<response>") :].strip()
        return cleaned

    def _retrieved_response(self, prompt: str) -> str | None:
        """Return a curated answer when the final user request closely matches one."""
        user_request = self._extract_user_request(prompt)
        match = self._instruction_retriever.retrieve(user_request)
        if match is None:
            return None
        return match.example.response.strip()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    ) -> str:
        """Generate an assistant completion, using KV-cached decoding."""
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
            raise ValueError("no_repeat_ngram_size must be non-negative")

        retrieved = self._retrieved_response(prompt)
        if retrieved is not None:
            return retrieved

        prompt_ids = self._prompt_ids(prompt)
        if len(prompt_ids) > self.model.block_size:
            prompt_ids = prompt_ids[-self.model.block_size :]

        generated_ids = list(prompt_ids)
        completion_ids: list[int] = []
        eos_id = self.tokenizer.stoi["<eos>"]

        context = torch.tensor([prompt_ids], dtype=torch.long)
        logits, past_key_values = self.model.forward_cached(context)
        next_logits = logits[:, -1, :]

        available_tokens = max(0, self.model.block_size - len(prompt_ids))
        generation_limit = min(max_new_tokens, available_tokens)

        for _ in range(generation_limit):
            next_logits = self._apply_repetition_penalty(
                next_logits,
                completion_ids,
                repetition_penalty,
            )
            next_logits = self._block_repeated_ngram(
                next_logits,
                completion_ids,
                no_repeat_ngram_size,
            )
            next_id = self._select_next_token(next_logits, temperature)

            generated_ids.append(next_id)
            completion_ids.append(next_id)
            if next_id == eos_id:
                break
            if len(generated_ids) >= self.model.block_size:
                break

            next_token = torch.tensor([[next_id]], dtype=torch.long)
            logits, past_key_values = self.model.forward_cached(
                next_token,
                past_key_values,
            )
            next_logits = logits[:, -1, :]

        return self._clean_completion(self.tokenizer.decode(completion_ids))
