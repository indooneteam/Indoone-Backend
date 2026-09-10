from __future__ import annotations

from pathlib import Path

import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


class LocalModelRuntime:
    """Loads an Indoone checkpoint and generates text locally."""

    def __init__(self, checkpoint_path: Path, tokenizer_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        config = dict(checkpoint["config"])
        config.pop("model_version", None)
        self.model = IndooneTransformer(vocab_size=self.tokenizer.vocab_size, **config)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        generated_ids = list(prompt_ids)
        idx = torch.tensor([generated_ids], dtype=torch.long)
        for _ in range(max_new_tokens):
            context = idx[:, -self.model.block_size :]
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            generated_ids.append(next_id.item())
            if next_id.item() == self.tokenizer.stoi["<eos>"]:
                break

        completion_ids = generated_ids[len(prompt_ids) :]
        return self.tokenizer.decode(completion_ids).strip()
