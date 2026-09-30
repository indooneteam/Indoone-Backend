"""Local Qwen3-0.6B service for the Indoone terminal backend.

The terminal backend uses a locally downloaded GGUF model. The model is loaded
from the local filesystem and is never required at application startup.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading

from app.ai.qwen_runtime import QwenLocalModelRuntime, QwenRuntimeError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "qwen3-0.6b"

MODEL_DIR = Path(os.getenv("QWEN_MODEL_DIR", str(_DEFAULT_MODEL_DIR))).expanduser()
MODEL_FILENAME = os.getenv("QWEN_MODEL_FILENAME", "Qwen3-0.6B-Q4_0.gguf").strip()
_MODEL_PATH = Path(
    os.getenv("QWEN_MODEL_PATH", str(MODEL_DIR / MODEL_FILENAME))
).expanduser()

_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_RUNTIME: QwenLocalModelRuntime | None = None


def _ensure_model() -> None:
    if _MODEL_PATH.is_file():
        return
    raise QwenRuntimeError(
        f"Qwen local model file not found: {_MODEL_PATH}. "
        "Place the verified Qwen3-0.6B GGUF model at this path before starting AI chat."
    )


def _load_runtime() -> QwenLocalModelRuntime:
    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    with _LOAD_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME

        _ensure_model()

        try:
            context_size = int(os.getenv("QWEN_CONTEXT_SIZE", "1024"))
            threads = int(os.getenv("QWEN_THREADS", "2"))
        except ValueError as exc:
            raise RuntimeError("Qwen context/threads configuration is invalid") from exc

        _RUNTIME = QwenLocalModelRuntime(
            _MODEL_PATH,
            context_size=context_size,
            threads=threads,
        )
        return _RUNTIME


async def generate_qwen_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    messages: list[dict[str, str]] = []
    if history:
        for role, content in history[-12:]:
            clean_content = " ".join(str(content).split()).strip()
            if clean_content and role in {"user", "assistant"}:
                messages.append({"role": role, "content": clean_content[:4_000]})

    user_content = cleaned
    if document_context.strip():
        user_content += "\n\nUSER-PROVIDED DOCUMENT:\n" + document_context.strip()[:100_000]
    messages.append({"role": "user", "content": user_content})

    runtime = await asyncio.to_thread(_load_runtime)

    try:
        max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "256"))
    except ValueError as exc:
        raise RuntimeError("QWEN_MAX_TOKENS configuration is invalid") from exc

    with _INFERENCE_LOCK:
        return await asyncio.to_thread(
            runtime.generate,
            messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
        )
