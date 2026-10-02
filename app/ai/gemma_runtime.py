from __future__ import annotations

from pathlib import Path
import os
import threading

from llama_cpp import Llama


class GemmaRuntimeError(RuntimeError):
    """Raised when the local Gemma runtime cannot be loaded or used."""


class GemmaLocalModelRuntime:
    """CPU-first runtime for a local Gemma 3 GGUF model."""

    def __init__(
        self,
        model_path: Path,
        *,
        context_size: int = 1024,
        threads: int | None = None,
    ) -> None:
        if not model_path.is_file():
            raise GemmaRuntimeError(f"Gemma model file not found: {model_path}")

        cpu_threads = threads or min(2, os.cpu_count() or 1)

        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=cpu_threads,
            n_batch=64,
            n_gpu_layers=0,
            chat_format=None,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.2,
    ) -> str:
        if not messages:
            raise ValueError("messages cannot be empty")

        with self._lock:
            try:
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                raise GemmaRuntimeError("Gemma generation failed") from exc

        choices = result.get("choices") or []
        if not choices:
            raise GemmaRuntimeError("Gemma returned no choices")

        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise GemmaRuntimeError("Gemma returned invalid response")

        return content.strip()
