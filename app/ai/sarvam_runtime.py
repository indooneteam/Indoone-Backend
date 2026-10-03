from __future__ import annotations

from pathlib import Path
import os
import threading

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - optional until candidate runtime is enabled
    Llama = None


class SarvamRuntimeError(RuntimeError):
    """Raised when the local Sarvam candidate cannot be loaded or used."""


def _render_prompt(messages: list[dict[str, str]]) -> str:
    """Render Sarvam-1 messages using the verified Llama-style [INST] format."""
    if not messages:
        raise ValueError("messages cannot be empty")

    normalized = [dict(message) for message in messages]
    for message in normalized:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(
                "Sarvam supports only system, user, and assistant messages"
            )

    system_text = ""
    start_index = 0
    if normalized[0].get("role") == "system":
        system_text = str(normalized[0].get("content", "")).strip()
        start_index = 1

    rendered_parts: list[str] = []
    first_user = True

    for message in normalized[start_index:]:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        if role == "user":
            if first_user and system_text:
                content = f"<<SYS>>\n{system_text}\n<</SYS>>\n{content}"
            rendered_parts.append(f"[INST] {content} [/INST]")
            first_user = False
        elif role == "assistant":
            rendered_parts.append(f" {content} </s>")

    prompt = "".join(rendered_parts).strip()
    if not prompt:
        raise ValueError("Sarvam prompt is empty")
    return prompt


class SarvamLocalModelRuntime:
    """CPU-first runtime for the tested Sarvam-1 GGUF model."""

    def __init__(
        self,
        model_path: Path,
        *,
        context_size: int = 2048,
        threads: int | None = None,
    ) -> None:
        if Llama is None:
            raise SarvamRuntimeError(
                "llama-cpp-python is not installed; install the CPU wheel before using Sarvam"
            )
        if not model_path.is_file():
            raise SarvamRuntimeError(f"Sarvam model file not found: {model_path}")
        if context_size <= 0:
            raise ValueError("context_size must be greater than zero")

        detected_threads = os.cpu_count() or 1
        cpu_threads = threads or min(2, detected_threads)
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
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> str:
        if not messages:
            raise ValueError("messages cannot be empty")
        if max_tokens < 0:
            raise ValueError("max_tokens must not be negative")

        prompt = _render_prompt(messages)

        with self._lock:
            try:
                result = self._llm.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=["</s>", "[INST]"],
                )
            except Exception as exc:
                raise SarvamRuntimeError("Sarvam local generation failed") from exc

        choices = result.get("choices") or []
        if not choices:
            raise SarvamRuntimeError("Sarvam returned no choices")

        content = choices[0].get("text", "")
        if not isinstance(content, str):
            raise SarvamRuntimeError("Sarvam returned an invalid response")

        return content.strip()
