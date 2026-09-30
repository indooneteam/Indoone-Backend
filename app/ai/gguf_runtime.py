from __future__ import annotations

from pathlib import Path
import os
import threading

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover
    Llama = None


class GGUFModelRuntimeError(RuntimeError):
    """Raised when a local GGUF model cannot be loaded or used."""


class GGUFModelRuntime:
    """Generic CPU-first llama.cpp runtime for a local chat/instruct GGUF."""

    def __init__(
        self,
        model_path: Path,
        *,
        context_size: int = 512,
        threads: int | None = None,
    ) -> None:
        if Llama is None:
            raise GGUFModelRuntimeError("llama-cpp-python is not installed")
        if not model_path.is_file():
            raise GGUFModelRuntimeError(f"GGUF model file not found: {model_path}")
        if context_size <= 0:
            raise ValueError("context_size must be greater than zero")

        cpu_threads = threads or min(2, os.cpu_count() or 1)
        if cpu_threads <= 0:
            raise ValueError("threads must be greater than zero")

        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=cpu_threads,
            n_batch=min(64, context_size),
            n_gpu_layers=0,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> str:
        if not messages:
            raise ValueError("messages cannot be empty")
        with self._lock:
            try:
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=1.1,
                )
            except Exception as exc:
                raise GGUFModelRuntimeError("GGUF local generation failed") from exc

        choices = result.get("choices") or []
        if not choices:
            raise GGUFModelRuntimeError("GGUF returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise GGUFModelRuntimeError("GGUF returned an invalid response")
        return content.strip()
