"""Local Sarvam-1 service for the Indoone terminal backend.

The terminal backend uses a locally downloaded GGUF model. Sarvam is a
candidate backend selected explicitly through INDOONE_MODEL_BACKEND=sarvam.
The runtime is loaded lazily and never required at application startup.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading

from app.ai.sarvam_runtime import SarvamLocalModelRuntime, SarvamRuntimeError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "sarvam-1" / "sarvam-1-Q4_K_M.gguf"

_MODEL_PATH = Path(
    os.getenv("SARVAM_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
).expanduser()

_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_RUNTIME: SarvamLocalModelRuntime | None = None


def _load_runtime() -> SarvamLocalModelRuntime:
    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    with _LOAD_LOCK:
        if _RUNTIME is None:
            try:
                context_size = int(os.getenv("SARVAM_CONTEXT_SIZE", "2048"))
                threads = int(os.getenv("SARVAM_THREADS", "2"))
                if context_size <= 0 or threads <= 0:
                    raise ValueError
            except ValueError as exc:
                raise SarvamRuntimeError(
                    "Sarvam context/threads configuration is invalid"
                ) from exc

            _RUNTIME = SarvamLocalModelRuntime(
                _MODEL_PATH,
                context_size=context_size,
                threads=threads,
            )

    return _RUNTIME


async def generate_sarvam_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are Indoone AI, a helpful assistant. "
                "Answer the latest user message directly and use earlier turns only as conversation context. "
                "Answer in the same language as the latest user message. "
                "Do not repeat the user's question and never output role labels such as User: or Assistant:. "
                "For Romanized Kannada, understand it as Kannada and answer in natural Kannada; "
                "Roman Kannada is acceptable when the user writes Kannada in English letters, but do not switch to English. "
                "When live web evidence is provided, use that evidence for factual or current claims, prefer the most relevant "
                "and recent evidence, and do not invent facts that are not supported by the evidence."
            ),
        }
    ]

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
        max_tokens = int(os.getenv("SARVAM_MAX_TOKENS", "256"))
    except ValueError as exc:
        raise RuntimeError("SARVAM_MAX_TOKENS configuration is invalid") from exc

    with _INFERENCE_LOCK:
        reply = await asyncio.to_thread(
            runtime.generate,
            messages,
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
        )

    if not reply:
        raise RuntimeError("Sarvam returned an empty user-facing answer")

    return reply
