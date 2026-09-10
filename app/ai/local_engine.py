from __future__ import annotations

from pathlib import Path

import torch

from app.ai.model import IndooneTransformer
from app.ai.tokenizer import BPETokenizer


DEFAULT_CHECKPOINT = Path("models/indoone-small/indoone-small.pt")
DEFAULT_TOKENIZER = Path("models/indoone-small/tokenizer.json")


class LocalAIEngine:
    """Loads and runs an Indoone-owned local language model."""

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
            config = payload["config"]
            self.model = IndooneTransformer(
                vocab_size=self.tokenizer.vocab_size,
                **config,
            )
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
        except Exception as exc:
            self._load_error = f"failed to load Indoone model: {exc}"

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
            and self._load_error is None
        )

    @torch.inference_mode()
    async def generate(
        self,
        message: str,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
    ) -> str:
        if not self.ready:
            raise RuntimeError(self._load_error or "Indoone local model is unavailable")
        assert self.model is not None
        assert self.tokenizer is not None

        prompt = message.strip()
        if not prompt:
            raise ValueError("message cannot be empty")

        generated = self.tokenizer.encode(prompt)
        for _ in range(max_new_tokens):
            context = torch.tensor(
                [generated[-self.model.block_size :]], dtype=torch.long
            )
            logits, _ = self.model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probabilities = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_id)
            if next_id == self.tokenizer.stoi["<eos>"]:
                break
        return self.tokenizer.decode(generated)
