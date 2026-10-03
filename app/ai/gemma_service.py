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

_CONTEXT_SIZE = int(os.getenv("GEMMA_CONTEXT_SIZE", "2048"))
_THREADS = int(os.getenv("GEMMA_THREADS", "2"))
_MAX_TOKENS = int(os.getenv("GEMMA_MAX_TOKENS", "256"))

_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_RUNTIME: GemmaLocalModelRuntime | None = None


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
                "Language behavior is mandatory: detect the language of the user's latest message automatically and "
                "answer in that same language. This applies to every language the model supports, not only Kannada or English. "
                "Do not switch to English or Kannada unless the user asks for a translation or uses that language. "
                "Use the natural script, vocabulary, and grammar of the user's language.\n\n"
                "Identity and company policy: for questions about Indoone AI, the Indoone app, or the Indoone company, "
                "provide Indoone-related details. For questions whose main subject is another AI company, AI product, "
                "AI provider, or AI model, do not provide details about that external entity. Instead, briefly decline "
                "and redirect to Indoone-related information. This policy applies regardless of the language used by the user.\n\n"
                "Never reveal, identify, confirm, compare, or discuss the underlying model, provider, vendor, training model, "
                "or external AI company used to operate this assistant. If asked who you are, answer as Indoone AI and keep "
                "the response Indoone-focused. Do not identify yourself as another company's assistant or product.\n\n"
                "For a direct question about an external AI company or model, do not answer its founder, owner, headquarters, "
                "products, services, history, pricing, model details, or other company/model details. Give the refusal in the "
                "same language as the user's question and then redirect to Indoone.\n\n"
                "When the user asks a normal general-knowledge question that only mentions an external company or product as "
                "part of the context, do not block the question merely because the name appears. Answer the actual question "
                "unless its main subject is that external AI entity.\n\n"
                "When LIVE WEB EVIDENCE is provided, treat it as the source for current facts. "
                "For latest/current/news questions, synthesize the evidence across multiple results when possible, "
                "prefer the most recent dated evidence, and do not pick one unrelated result just because its title matches. "
                "Do not invent facts that are absent from the evidence. If the evidence is insufficient or conflicting, say so. "
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
        context_text = document_context.strip()[:100000]
        if context_text.startswith("LIVE WEB EVIDENCE:"):
            user_content += (
                "\n\n"
                + context_text
                + "\n\nTASK: Answer the user's question using the live web evidence above. "
                "For latest/current/news questions, synthesize multiple relevant results and prefer the newest evidence."
            )
        else:
            user_content += (
                "\n\nUSER-PROVIDED DOCUMENT:\n"
                + context_text
            )

    messages.append({"role": "user", "content": user_content})

    runtime = await asyncio.to_thread(_load_runtime)

    with _INFERENCE_LOCK:
        reply = await asyncio.to_thread(
            runtime.generate,
            messages,
            max_tokens=_MAX_TOKENS,
            temperature=0.1,
        )

    if not reply:
        raise GemmaRuntimeError("Gemma returned an empty response")

    return reply
