from __future__ import annotations

from pathlib import Path
import re

import torch

from app.ai.language_detection import SCRIPT_RANGES

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer
from app.ai.training.training_data import format_instruction_prompt

DEFAULT_MAX_NEW_TOKENS = 192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = 1.08
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_TOP_K = 64
MIN_GENERATED_TOKENS_BEFORE_EOS = 4

class LocalModelRuntime:
    """Load the trained Indoone local language model and generate model output."""

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

        self._language_token_ids: dict[str, frozenset[int]] = {}

    @staticmethod
    def _select_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int = DEFAULT_TOP_K,
    ) -> int:
        """Select the next token with greedy or bounded sampling."""
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        if temperature == 0:
            return int(torch.argmax(logits, dim=-1).item())

        if logits.ndim != 2 or logits.size(0) != 1:
            raise ValueError("logits must have shape (1, vocab_size)")

        limit = min(top_k, logits.size(-1)) if top_k else logits.size(-1)
        top_values, top_indices = torch.topk(logits, k=limit, dim=-1)
        probabilities = torch.softmax(top_values / temperature, dim=-1)
        choice = torch.multinomial(probabilities, num_samples=1)
        return int(top_indices.gather(1, choice).item())

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

    def _prompt_ids(self, prompt: str) -> list[int]:
        """Render a user request in the same instruction format used for training."""
        stripped_prompt = prompt.strip()
        if stripped_prompt.startswith("<instruction>") and stripped_prompt.endswith("<response>"):
            formatted_prompt = stripped_prompt
        else:
            trailing_response = re.fullmatch(
                r"(?s)(<instruction>.*</instruction>\s*<response>)",
                stripped_prompt,
            )
            formatted_prompt = trailing_response.group(1) if trailing_response else format_instruction_prompt(stripped_prompt)
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

    def _allowed_token_ids_for_language(self, language: str) -> frozenset[int]:
        """Cache token ids that can safely contribute to the requested script."""
        normalized = language.strip().casefold() or "english"
        cached = self._language_token_ids.get(normalized)
        if cached is not None:
            return cached

        script_pattern = None if normalized == "english" else dict(SCRIPT_RANGES).get(language)
        allowed: set[int] = set()
        special_ids = {
            self.tokenizer.stoi[token]
            for token in self.tokenizer.SPECIAL_TOKENS
            if token in self.tokenizer.stoi
        }
        for token_id in range(self.tokenizer.vocab_size):
            if token_id in special_ids:
                allowed.add(token_id)
                continue
            try:
                decoded = self.tokenizer.decode([token_id])
            except Exception:
                continue
            if not decoded:
                continue

            if normalized == "english":
                if re.search(r"[A-Za-z0-9]", decoded) or all(ord(char) < 128 for char in decoded):
                    allowed.add(token_id)
                continue

            if re.search(r"[A-Za-z0-9]", decoded):
                allowed.add(token_id)
                continue
            if script_pattern is not None and script_pattern.search(decoded):
                allowed.add(token_id)
                continue
            if all(ord(char) < 128 for char in decoded):
                allowed.add(token_id)

        result = frozenset(allowed)
        self._language_token_ids[normalized] = result
        return result

    def _apply_language_constraint(
        self,
        logits: torch.Tensor,
        language: str,
    ) -> torch.Tensor:
        """Prevent decoding from drifting into an unrelated writing script."""
        if not language.strip() or logits.ndim != 2 or logits.size(0) != 1:
            return logits
        allowed = self._allowed_token_ids_for_language(language)
        if not allowed:
            return logits
        constrained = logits.clone()
        blocked = [index for index in range(logits.size(-1)) if index not in allowed]
        if blocked and len(blocked) < logits.size(-1):
            constrained[0, blocked] = float("-inf")
            if not torch.isfinite(constrained).any():
                return logits
        return constrained

    @staticmethod
    def _is_email_request(prompt: str) -> bool:
        """Detect email-focused requests so the capability adapter is isolated."""
        normalized = " ".join(prompt.casefold().split())
        return bool(
            re.search(
                r"(?:\bemail\b|\be-mail\b|\binbox\b|\bmailbox\b|"
                r"\bphishing\b|\bspam\b|\bunsubscribe\b|"
                r"\bsubject\s*:|\bfrom\s*:|\bto\s*:)",
                normalized,
            )
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        language: str = "English",
        use_email_adapter: bool | None = None,
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
        if use_email_adapter is None:
            use_email_adapter = self._is_email_request(prompt)

        prompt_ids = self._prompt_ids(prompt)
        if len(prompt_ids) > self.model.block_size:
            prompt_ids = prompt_ids[-self.model.block_size :]

        generated_ids = list(prompt_ids)
        completion_ids: list[int] = []
        eos_id = self.tokenizer.stoi["<eos>"]

        context = torch.tensor([prompt_ids], dtype=torch.long)
        logits, past_key_values = self.model.forward_cached(
            context,
            use_email_adapter=use_email_adapter,
        )
        next_logits = logits[:, -1, :]

        available_tokens = max(0, self.model.block_size - len(prompt_ids))
        generation_limit = min(max_new_tokens, available_tokens)

        for _ in range(generation_limit):
            next_logits = self._apply_repetition_penalty(
                next_logits,
                completion_ids,
                repetition_penalty,
            )
            next_logits = self._apply_language_constraint(next_logits, language)
            next_logits = self._block_repeated_ngram(
                next_logits,
                completion_ids,
                no_repeat_ngram_size,
            )

            if len(completion_ids) < MIN_GENERATED_TOKENS_BEFORE_EOS:
                eos_blocked = next_logits.clone()
                eos_blocked[0, eos_id] = float("-inf")
                if torch.isfinite(eos_blocked).any():
                    next_logits = eos_blocked

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
                use_email_adapter=use_email_adapter,
            )
            next_logits = logits[:, -1, :]

        return self._clean_completion(self.tokenizer.decode(completion_ids))
