from __future__ import annotations

import asyncio
import os
from pathlib import Path
import threading

from app.ai.gemma_runtime import GemmaLocalModelRuntime, GemmaRuntimeError

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MODEL_PATH = Path(
    os.getenv(
        "GEMMA_MODEL_PATH",
        str(_PROJECT_ROOT / "models" / "gemma3-4b-it" / "gemma-3-4b-it-Q4_K_M.gguf"),
    )
).expanduser()

_CONTEXT_SIZE = int(os.getenv("GEMMA_CONTEXT_SIZE", "1024"))
_THREADS = int(os.getenv("GEMMA_THREADS", "2"))
_MAX_TOKENS = int(os.getenv("GEMMA_MAX_TOKENS", "256"))

_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_RUNTIME: GemmaLocalModelRuntime | None = None


def _identity_response(message: str) -> str | None:
    normalized = " ".join(message.casefold().split())

    identity_hints = (
        # Kannada identity questions
        "ನಿನ್ನ ಬಗ್ಗೆ",
        "ನಿಮ್ಮ ಬಗ್ಗೆ",
        "ನೀನು ಯಾರು",
        "ನೀವು ಯಾರು",
        "ಯಾರು ನೀನು",
        "ಯಾರು ನೀವು",
        "ನಿನ್ನ ಹೆಸರು",
        "ನಿಮ್ಮ ಹೆಸರು",
        "ನಿನ್ನನ್ನು ಯಾರು",
        "ನಿಮ್ಮನ್ನು ಯಾರು",
        "ನಿನ್ನನ್ನು ಮಾಡಿದ್ದು ಯಾರು",
        "ನಿಮ್ಮನ್ನು ಮಾಡಿದ್ದು ಯಾರು",
        "ಯಾರು ತಯಾರಿಸಿದರು",
        "ಯಾರು ಅಭಿವೃದ್ಧಿಪಡಿಸಿದರು",
        "ಯಾವ ಮಾಡೆಲ್",
        "ಯಾವ ಎಐ ಮಾಡೆಲ್",
        # English identity questions
        "who are you",
        "what are you",
        "tell me about yourself",
        "your name",
        "who made you",
        "who created you",
        "who developed you",
        "which model are you",
        "what model are you",
        "which ai model",
        "what ai model",
        "what is your model",
        "who built you",
        "who is your creator",
        "where are you from",
    )

    if any(hint in normalized for hint in identity_hints):
        if any(token in normalized for token in ("ನ", "ಕನ್ನಡ", "ನಿನ್ನ", "ನಿಮ್ಮ")):
            return (
                "ನಾನು Indoone AI. ನಾನು Indoone appನ AI assistant. "
                "ನಾನು Indooneಗಾಗಿ ವಿನ್ಯಾಸಗೊಳಿಸಲಾದ ಸಹಾಯಕ."
            )
        return (
            "I’m Indoone AI, the AI assistant for the Indoone app. "
            "I’m designed specifically for Indoone."
        )

    return None

def _load_runtime() -> GemmaLocalModelRuntime:
    global _RUNTIME

    if _RUNTIME is not None:
        return _RUNTIME

    with _LOAD_LOCK:
        if _RUNTIME is None:
            _RUNTIME = GemmaLocalModelRuntime(
                _MODEL_PATH,
                context_size=_CONTEXT_SIZE,
                threads=_THREADS,
            )

    return _RUNTIME


async def generate_gemma_reply(
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
                "You are Indoone AI, the AI assistant for the Indoone app. "
                "For identity, creator, developer, provider, or model questions, keep the answer about Indoone AI only. "
                "Do not disclose, name, or discuss any underlying model, provider, vendor, training model, or external AI company. "
                "Do not identify yourself as another company's assistant or product. "
                "Answer the user in the same language as the user's question. "
                "For Kannada questions, use natural Kannada script. "
                "For English questions, use natural English. "
                "Be accurate, helpful, friendly, and concise."
            ),
        }
    ]

    if history:
        for role, content in history[-12:]:
            content = " ".join(str(content).split()).strip()
            if content and role in {"user", "assistant"}:
                messages.append(
                    {"role": role, "content": content[:4000]}
                )

    user_content = cleaned
    if document_context.strip():
        user_content += (
            "\n\nUSER-PROVIDED DOCUMENT:\n"
            + document_context.strip()[:100000]
        )

    identity_reply = _identity_response(cleaned)
    if identity_reply is not None:
        return identity_reply

    messages.append({"role": "user", "content": user_content})

    runtime = await asyncio.to_thread(_load_runtime)

    with _INFERENCE_LOCK:
        reply = await asyncio.to_thread(
            runtime.generate,
            messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.2,
        )

    if not reply:
        raise GemmaRuntimeError("Gemma returned an empty response")

    return reply
