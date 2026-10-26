"""Indoone AI core entry point.

Local Indoone inference remains the default. Gemini chat can be enabled explicitly
with INDOONE_MODEL_BACKEND=gemini; supported Gemma 4 models can then use the
bounded free web-research tool when the question calls for current information.
"""

from datetime import datetime, timedelta, timezone
import asyncio
from pathlib import Path
import logging
import os
import threading
from typing import TYPE_CHECKING
import time

from app.ai.language.language_detection import detect_response_language
from app.storage.github_release import GitHubReleaseStorageError, get_github_release_storage


logger = logging.getLogger(__name__)


def _gemini_backend_enabled() -> bool:
    """Use Gemini only when explicitly selected; preserve local inference by default."""
    return os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "gemini"


def _azure_backend_enabled() -> bool:
    """Use Azure OpenAI as the primary provider when explicitly selected."""
    return os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "azure"


def ai_provider_status() -> dict[str, object]:
    """Return safe provider/model configuration details for deploy readiness.

    Never include the API key value. The Gemini model ID is safe to report and
    makes it possible to verify that the deployed chat routes to the intended model.
    """
    if _azure_backend_enabled():
        from app.ai.gateway import azure_openai_service
        return {
            "provider": "azure",
            "model": azure_openai_service.azure_model_name(),
            "configuration_ready": azure_openai_service.azure_configuration_ready(),
            "api_key_configured": bool(os.getenv("AZURE_OPENAI_API_KEY", "").strip()),
        }

    if not _gemini_backend_enabled():
        return {
            "provider": "local",
            "model": None,
            "configuration_ready": True,
            "api_key_configured": None,
        }

    from app.ai.gateway import gemini_service
    model = os.getenv("GEMINI_MODEL", gemini_service._DEFAULT_MODEL).strip()
    api_key_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return {
        "provider": "gemini",
        "model": model or None,
        "configuration_ready": bool(api_key_configured and model),
        "api_key_configured": api_key_configured,
    }


if TYPE_CHECKING:
    from app.ai.local.inference import LocalModelRuntime

# Kept as a lazy injection point for tests and compatibility. The real class is
# imported only inside _load_local_model_runtime() so Render startup stays light.
LocalModelRuntime = None

MODEL_DIR = Path("models/indoone-small")
_checkpoint = MODEL_DIR / "indoone-small.pt"
_tokenizer = MODEL_DIR / "tokenizer.json"
# Keep model loading entirely request-driven. Constructing LocalAIEngine at import
# time can eagerly load the full PyTorch checkpoint during Render startup, which
# increases memory pressure and can cause the web process to restart before chat.
_runtime: "LocalModelRuntime | None" = None
_NEXT_MODEL_LOAD_ATTEMPT = 0.0
_MODEL_LOAD_RETRY_SECONDS = 60.0
_MODEL_MAX_NEW_TOKENS = 512
# Serialize the artifact check/runtime load so concurrent chat requests cannot
# trigger duplicate B2 downloads or duplicate model loads.
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_INFERENCE_LOCK = threading.Lock()
_ARTIFACT_CHECK_COMPLETED = False


def _ensure_model_artifacts(model_dir: Path) -> None:
    """Load model artifacts only from the configured private Indoone-Model release."""
    github_storage = get_github_release_storage()
    if github_storage is None:
        raise GitHubReleaseStorageError(
            "Indoone model serving requires GITHUB_MODEL_REPOSITORY, "
            "GITHUB_MODEL_RELEASE_TAG, and GITHUB_MODEL_TOKEN"
        )

    artifacts = {
        "indoone-small.pt": model_dir / "indoone-small.pt",
        "tokenizer.json": model_dir / "tokenizer.json",
    }
    for filename, local_path in artifacts.items():
        if local_path.exists():
            continue
        github_storage.download_file(filename, local_path)


def _next_utc_midnight_timestamp() -> float:
    """Back off failed remote model-artifact downloads until the next UTC day."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).date()
    midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    return midnight.timestamp()


def _load_local_model_runtime() -> "LocalModelRuntime | None":
    """Load the trained checkpoint with a single-flight artifact download/load."""
    global _runtime, _NEXT_MODEL_LOAD_ATTEMPT, _ARTIFACT_CHECK_COMPLETED

    if _runtime is not None:
        return _runtime

    if not _MODEL_LOAD_LOCK.acquire(blocking=False):
        logger.info("Indoone model load already in progress; another request will reuse the loaded runtime")
        return None

    try:
        # A concurrent request may have completed the model load while this
        # request was waiting for the lock.
        if _runtime is not None:
            return _runtime

        now = time.time()
        if now < _NEXT_MODEL_LOAD_ATTEMPT:
            return None

        if not _ARTIFACT_CHECK_COMPLETED:
            try:
                _ensure_model_artifacts(MODEL_DIR)
                if get_github_release_storage() is not None:
                    logger.info("Indoone model artifacts checked from private GitHub Release")
                else:
                    logger.info("Indoone model artifact check completed from private Indoone-Model GitHub Release")
            except GitHubReleaseStorageError as exc:
                _NEXT_MODEL_LOAD_ATTEMPT = _next_utc_midnight_timestamp()
                logger.error("Indoone model artifact check failed: %s", exc)
                return None

            if not (_checkpoint.exists() and _tokenizer.exists()):
                _NEXT_MODEL_LOAD_ATTEMPT = _next_utc_midnight_timestamp()
                logger.error(
                    "Indoone local model artifacts are missing: checkpoint=%s tokenizer=%s",
                    _checkpoint.exists(), _tokenizer.exists(),
                )
                return None

            # From this point on, every request in this process uses the local
            # artifacts and never re-checks/downloads them from remote storage.
            _ARTIFACT_CHECK_COMPLETED = True

        try:
            # Import PyTorch/model code only when a chat request actually needs
            # trained-model inference. This keeps Render startup memory low while
            # retaining a testable lazy injection point.
            runtime_factory = LocalModelRuntime
            if runtime_factory is None:
                from app.ai.local.inference import LocalModelRuntime as runtime_factory

            _runtime = runtime_factory(_checkpoint, _tokenizer)
            _NEXT_MODEL_LOAD_ATTEMPT = 0.0
            logger.info("Indoone local model runtime loaded successfully")
        except Exception as exc:
            _NEXT_MODEL_LOAD_ATTEMPT = now + _MODEL_LOAD_RETRY_SECONDS
            logger.error("Indoone local model runtime failed to load: %s", exc)
            _runtime = None
        return _runtime
    finally:
        _MODEL_LOAD_LOCK.release()


def _generate_with_local_model(
    runtime: "LocalModelRuntime",
    context: str,
    language: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    # Render Free provides a very small CPU budget. Serialize local inference so
    # concurrent requests cannot multiply the model's CPU/memory pressure.
    with _MODEL_INFERENCE_LOCK:
        return runtime.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            language=language,
        )


class _LocalModelAnswerProvider:
    """Generate every user-facing answer with the trained Indoone model."""

    async def generate(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = _MODEL_MAX_NEW_TOKENS,
    ) -> str:
        runtime = _load_local_model_runtime()
        if runtime is None:
            raise RuntimeError("trained Indoone local model is unavailable")

        context = f"{system_instruction.strip()}\n\n{user_prompt.strip()}".strip()
        try:
            return await asyncio.to_thread(
                _generate_with_local_model,
                runtime,
                context,
                detect_response_language(user_prompt),
                min(temperature, 0.7),
                min(max_output_tokens, _MODEL_MAX_NEW_TOKENS),
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("trained Indoone local model generation failed") from exc


_model_answer_provider = _LocalModelAnswerProvider()


class LocalAIService:
    """Configured AI answer entry point; local inference remains the default."""

    async def generate(
        self,
        message: str,
        history: list[tuple[str, str]] | None = None,
        document_context: str = "",
        media_attachment: dict[str, object] | None = None,
        media_attachments: list[dict[str, object]] | None = None,
        extra_system_instruction: str = "",
    ) -> str:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("message cannot be empty")

        # Keep the existing local-model path as the default. When explicitly configured,
        # Google Gemini handles the chat turn and can call the bounded free web-search tool.
        if _gemini_backend_enabled() or _azure_backend_enabled():
            from app.ai.gateway.gemini_service import generate_gemini_reply, run_with_provider_fallback
            from app.ai.gateway.azure_openai_service import generate_azure_reply

            async def _reply_with(provider):
                if media_attachments:
                    return await provider(
                        cleaned_message,
                        history=history,
                        document_context=document_context,
                        media_attachments=media_attachments,
                        extra_system_instruction=extra_system_instruction,
                    )
                if media_attachment is not None:
                    return await provider(
                        cleaned_message,
                        history=history,
                        document_context=document_context,
                        media_attachment=media_attachment,
                        extra_system_instruction=extra_system_instruction,
                    )
                return await provider(
                    cleaned_message,
                    history=history,
                    document_context=document_context,
                    extra_system_instruction=extra_system_instruction,
                )

            _azure_first = _azure_backend_enabled()
            return await run_with_provider_fallback(
                lambda: _reply_with(generate_azure_reply if _azure_first else generate_gemini_reply),
                lambda: _reply_with(generate_gemini_reply if _azure_first else generate_azure_reply),
            )

        if media_attachment is not None or media_attachments:
            raise RuntimeError(
                "multimodal attachment processing requires INDOONE_MODEL_BACKEND=gemini or =azure"
            )

        prompt_parts = [cleaned_message]

        if history:
            history_lines = []
            for role, content in history[-12:]:
                clean_content = " ".join(str(content).split()).strip()
                if clean_content:
                    history_lines.append(f"{role}: {clean_content[:4_000]}")
            if history_lines:
                prompt_parts.extend(["", "CONVERSATION HISTORY:", *history_lines])

        if document_context.strip():
            prompt_parts.extend(["", "USER-PROVIDED DOCUMENT:", document_context.strip()[:100_000]])

        if _model_answer_provider is None:
            raise RuntimeError("trained Indoone model provider is unavailable")
        # Keep inference on the same prompt distribution used during SFT:
        # <instruction> contains the user's request directly, without injecting
        # a long runtime policy/system prefix that was not part of training.
        return await _model_answer_provider.generate(
            system_instruction="",
            user_prompt="\n".join(prompt_parts),
            temperature=0.0,
            max_output_tokens=_MODEL_MAX_NEW_TOKENS,
        )

async def stream_generate(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
    extra_system_instruction: str = "",
):
    """Yield the answer as it is produced, instead of returning it whole.

    The contract is a plain async generator of text chunks. Two things matter about it:

    * A provider that cannot stream still works. It yields the finished reply as ONE chunk,
      which is exactly what the old 48-character replay pretended to be - but this says so
      instead of chopping a complete string to look busy.
    * A streaming failure mid-answer is not silent. Anything the provider raises after the
      first chunk propagates, so the caller can tell the difference between "the answer is
      long" and "the connection died".

    Provider streaming is behind INDOONE_STREAM_PROVIDERS because it cannot be verified
    from a test suite: it is a live HTTP contract with Gemini and Azure. Until that is
    verified on a deployment, this returns the blocking reply as a single chunk - correct,
    just not incremental.
    """
    if os.getenv("INDOONE_STREAM_PROVIDERS", "").strip().casefold() in {"1", "true", "yes"}:
        provider_stream = _provider_stream()

        if provider_stream is not None:
            async for chunk in provider_stream(
                message,
                history=history,
                document_context=document_context,
                extra_system_instruction=extra_system_instruction,
            ):
                yield chunk
            return

    reply = await generate_reply(
        message,
        history=history,
        document_context=document_context,
        extra_system_instruction=extra_system_instruction,
    )

    if reply:
        yield reply


def _provider_stream():
    """The streaming generator for the configured backend, or None.

    Chosen the same way the blocking call is chosen, so the two can never disagree about
    which provider is active.
    """
    # A provider that has no streaming generator yet must not break the request: the
    # caller falls back to one chunk of the finished reply, which is correct, just not
    # incremental.
    try:
        if _gemini_backend_enabled():
            from app.ai.gateway.gemini_service import stream_gemini_reply

            return stream_gemini_reply

        if _azure_backend_enabled():
            from app.ai.gateway.azure_openai_service import stream_azure_reply

            return stream_azure_reply
    except ImportError:
        return None

    return None


async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
    extra_system_instruction: str = "",
) -> str:
    if media_attachments:
        return await LocalAIService().generate(
            message,
            history=history,
            document_context=document_context,
            media_attachments=media_attachments,
            extra_system_instruction=extra_system_instruction,
        )
    if media_attachment is None:
        return await LocalAIService().generate(
            message,
            history=history,
            document_context=document_context,
            extra_system_instruction=extra_system_instruction,
        )
    return await LocalAIService().generate(
        message,
        history=history,
        document_context=document_context,
        media_attachment=media_attachment,
        extra_system_instruction=extra_system_instruction,
    )
