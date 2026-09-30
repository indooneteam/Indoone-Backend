from __future__ import annotations

from pathlib import Path
import os
import re
import threading

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional until candidate runtime is enabled
    Llama = None


class QwenRuntimeError(RuntimeError):
    """Raised when the local Qwen candidate cannot be loaded or used."""


class QwenLocalModelRuntime:
    """CPU-first runtime for an already-trained Qwen3 GGUF model."""

    def __init__(
        self,
        model_path: Path,
        *,
        context_size: int = 1024,
        threads: int | None = None,
    ) -> None:
        if Llama is None:
            raise QwenRuntimeError(
                "llama-cpp-python is not installed; install the CPU wheel before using Qwen"
            )
        if not model_path.is_file():
            raise QwenRuntimeError(f"Qwen model file not found: {model_path}")
        if context_size <= 0:
            raise ValueError("context_size must be greater than zero")

        detected_threads = os.cpu_count() or 1
        cpu_threads = threads or min(4, detected_threads)
        if cpu_threads <= 0:
            raise ValueError("threads must be greater than zero")

        self._lock = threading.Lock()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=context_size,
            n_threads=cpu_threads,
            n_batch=min(128, context_size),
            n_gpu_layers=0,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.5,
    ) -> str:
        if not messages:
            raise ValueError("messages cannot be empty")
        normalized_messages = [dict(message) for message in messages]
        if normalized_messages and normalized_messages[-1].get("role") == "user":
            user_content = str(normalized_messages[-1].get("content", "")).strip()
            if "/no_think" not in user_content:
                normalized_messages[-1]["content"] = f"{user_content} /no_think".strip()
        if max_tokens < 0:
            raise ValueError("max_tokens must not be negative")

        with self._lock:
            try:
                result = self._llm.create_chat_completion(
                    messages=normalized_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                )
            except Exception as exc:
                raise QwenRuntimeError("Qwen local generation failed") from exc

        choices = result.get("choices") or []
        if not choices:
            raise QwenRuntimeError("Qwen returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str):
            raise QwenRuntimeError("Qwen returned an invalid response")
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
