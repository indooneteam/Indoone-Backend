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
import re
import time

from app.storage.github_release import GitHubReleaseStorageError, get_github_release_storage


logger = logging.getLogger(__name__)


def _gemini_backend_enabled() -> bool:
    """Use Gemini only when explicitly selected; preserve local inference by default."""
    return os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "gemini"

if TYPE_CHECKING:
    from app.ai.inference import LocalModelRuntime

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
                from app.ai.inference import LocalModelRuntime as runtime_factory

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


_SCRIPT_RANGES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Kannada", re.compile(r"[\u0C80-\u0CFF]")),
    ("Telugu", re.compile(r"[\u0C00-\u0C7F]")),
    ("Tamil", re.compile(r"[\u0B80-\u0BFF]")),
    ("Malayalam", re.compile(r"[\u0D00-\u0D7F]")),
    ("Hindi", re.compile(r"[\u0900-\u097F]")),
    ("Bengali", re.compile(r"[\u0980-\u09FF]")),
    ("Gujarati", re.compile(r"[\u0A80-\u0AFF]")),
    ("Punjabi", re.compile(r"[\u0A00-\u0A7F]")),
    ("Odia", re.compile(r"[\u0B00-\u0B7F]")),
    ("Urdu", re.compile(r"[\u0600-\u06FF]")),
)

_LANGUAGE_NAMES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("kannada", "ಕನ್ನಡ", "ಕನ್ನಡದ")),
    ("Hindi", ("hindi", "हिंदी", "हिन्दी")),
    ("Telugu", ("telugu", "తెలుగు")),
    ("Tamil", ("tamil", "தமிழ்", "தமிழ")),
    ("Malayalam", ("malayalam", "മലയാളം")),
    ("Marathi", ("marathi", "मराठी")),
    ("Bengali", ("bengali", "bangla", "বাংলা", "বাঙালি")),
    ("Assamese", ("assamese", "অসমীয়া", "অসমিয়া")),
    ("Gujarati", ("gujarati", "ગુજરાતી")),
    ("Punjabi", ("punjabi", "ਪੰਜਾਬੀ")),
    ("Odia", ("odia", "oriya", "ଓଡ଼ିଆ", "ଓଡିଆ")),
    ("Urdu", ("urdu", "اردو")),
)

_ROMANIZED_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Kannada", ("bagge", "helu", "heLi", "maadu", "maadi", "madbeku", "madbekagutte", "enidu", "enu", "yenu", "yenide", "ivaga", "matte", "nanage", "nimage", "nanna", "namma", "ide", "illa", "agide", "beku")),
    ("Hindi", ("kya", "hai", "hain", "mujhe", "aap", "aapko", "batao", "bataiye", "kaise", "kaisa", "kahan", "kyun", "nahi", "abhi", "mera", "meri")),
    ("Telugu", ("enti", "ela", "cheppu", "cheppandi", "undi", "ledu", "nenu", "meeru", "naaku", "emiti", "enduku", "ippudu", "malli")),
    ("Tamil", ("enna", "epdi", "eppadi", "sollu", "sollunga", "irukku", "illa", "naan", "neenga", "enakku", "ippo", "yen")),
    ("Malayalam", ("entha", "engane", "parayu", "parayoo", "undu", "illa", "njan", "ningal", "enikku", "ippo")),
    ("Marathi", ("kay", "aahe", "mala", "tumhi", "sanga", "kasa", "kashi", "kuthe", "ka", "nahi", "aata")),
    ("Bengali", ("ki", "ache", "ami", "apni", "bolo", "bolun", "amar", "keno", "nei")),
    ("Assamese", ("ki", "ase", "moi", "tumi", "kobo", "mur", "kio", "nai")),
    ("Gujarati", ("shu", "che", "chhe", "mane", "tame", "kaho", "kem", "nathi", "maru")),
    ("Punjabi", ("menu", "mainu", "tusi", "daso", "kive", "kiwe", "nahi", "mera", "sanu")),
    ("Odia", ("kana", "achhi", "mu", "tame", "kahantu", "kemiti", "nahi", "ebe")),
    ("Urdu", ("kya", "hai", "mujhe", "aap", "batao", "bataiye", "kaise", "kyun", "nahi")),
)


def _detect_response_language(message: str) -> str:
    normalized = " ".join(message.casefold().split())
    for language, names in _LANGUAGE_NAMES:
        for name in names:
            candidate = name.casefold()
            if re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized):
                return language
    for language, pattern in _SCRIPT_RANGES:
        if pattern.search(message):
            return language
    scores: dict[str, int] = {}
    for language, hints in _ROMANIZED_HINTS:
        score = sum(
            1
            for hint in hints
            if re.search(rf"(?<!\w){re.escape(hint.casefold())}(?!\w)", normalized)
        )
        if score:
            scores[language] = score
    if scores:
        best_language, best_score = max(scores.items(), key=lambda item: item[1])
        tied = [language for language, score in scores.items() if score == best_score]
        if best_score >= 2 or len(tied) == 1:
            return best_language
    return "English"


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
                _detect_response_language(user_prompt),
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
    ) -> str:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("message cannot be empty")

        # Keep the existing local-model path as the default. When explicitly configured,
        # Google Gemini handles the chat turn and can call the bounded free web-search tool.
        if _gemini_backend_enabled():
            from app.ai.gemini_service import generate_gemini_reply

            return await generate_gemini_reply(
                cleaned_message,
                history=history,
                document_context=document_context,
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

async def generate_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
) -> str:
    return await LocalAIService().generate(
        message,
        history=history,
        document_context=document_context,
    )
