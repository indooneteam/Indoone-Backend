from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from typing import Any

import httpx

MAX_AUDIO_BYTES = 12_000_000
MAX_TEXT_LENGTH = 8_000
MAX_LANGUAGE_LENGTH = 32
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    provider: str
    model: str | None
    confidence: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SynthesisResult:
    audio_base64: str
    mime_type: str
    provider: str
    model: str | None
    sample_rate_hz: int | None
    metadata: dict[str, Any]


def _timeout() -> float:
    raw = os.getenv("INDOONE_VOICE_TIMEOUT", str(DEFAULT_TIMEOUT)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("INDOONE_VOICE_TIMEOUT must be a number") from exc
    if value <= 0 or value > 600:
        raise ValueError("INDOONE_VOICE_TIMEOUT must be between 0 and 600 seconds")
    return value


def _decode_audio(audio_base64: str) -> bytes:
    if not audio_base64.strip():
        raise ValueError("audio_base64 is required")
    try:
        audio = base64.b64decode(audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("audio_base64 must be valid base64") from exc
    if not audio:
        raise ValueError("audio payload is empty")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("audio payload is too large")
    return audio


def _auth_headers(token_env: str) -> dict[str, str]:
    token = os.getenv(token_env, "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _require_url(env_name: str) -> str:
    url = os.getenv(env_name, "").strip()
    if not url:
        raise RuntimeError(f"{env_name} is not configured")
    return url


def _coerce_optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, confidence))


async def transcribe_audio(
    audio_base64: str,
    *,
    mime_type: str = "audio/wav",
    language: str = "",
) -> TranscriptionResult:
    audio = _decode_audio(audio_base64)
    url = _require_url("INDOONE_STT_URL")
    language = language.strip()[:MAX_LANGUAGE_LENGTH]
    payload = {
        "audio_base64": base64.b64encode(audio).decode("ascii"),
        "mime_type": mime_type.strip()[:120] or "audio/wav",
        "language": language,
    }
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            response = await client.post(url, json=payload, headers=_auth_headers("INDOONE_STT_TOKEN"))
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"speech-to-text provider failed: {exc}") from exc

    text = data.get("text")
    if not isinstance(text, str):
        raise RuntimeError("speech-to-text provider returned no text")
    return TranscriptionResult(
        text=text.strip()[:MAX_TEXT_LENGTH],
        language=(str(data["language"])[:MAX_LANGUAGE_LENGTH] if data.get("language") else None),
        provider=str(data.get("provider") or "local"),
        model=(str(data["model"]) if data.get("model") else None),
        confidence=_coerce_optional_confidence(data.get("confidence")),
        metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
    )


async def synthesize_speech(
    text: str,
    *,
    language: str = "",
    voice: str = "",
    format: str = "wav",
    audio_format: str | None = None,
) -> SynthesisResult:
    text = text.strip()
    if not text:
        raise ValueError("text is required")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError("text is too long")
    url = _require_url("INDOONE_TTS_URL")
    selected_format = (audio_format if audio_format is not None else format).strip()[:16] or "wav"
    payload = {
        "text": text,
        "language": language.strip()[:MAX_LANGUAGE_LENGTH],
        "voice": voice.strip()[:128],
        "format": selected_format,
    }
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            response = await client.post(url, json=payload, headers=_auth_headers("INDOONE_TTS_TOKEN"))
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"text-to-speech provider failed: {exc}") from exc

    encoded = data.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        audio = data.get("audio")
        if isinstance(audio, list) and audio and isinstance(audio[0], str):
            encoded = audio[0]
    if not isinstance(encoded, str) or not encoded:
        raise RuntimeError("text-to-speech provider returned no audio")
    try:
        audio = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("text-to-speech provider returned invalid base64 audio") from exc
    if not audio:
        raise RuntimeError("text-to-speech provider returned empty audio")
    if len(audio) > MAX_AUDIO_BYTES:
        raise RuntimeError("text-to-speech provider returned too much audio")

    sample_rate = data.get("sample_rate_hz")
    try:
        sample_rate = int(sample_rate) if sample_rate is not None else None
    except (TypeError, ValueError):
        sample_rate = None

    return SynthesisResult(
        audio_base64=encoded,
        mime_type=str(data.get("mime_type") or "audio/wav"),
        provider=str(data.get("provider") or "local"),
        model=(str(data["model"]) if data.get("model") else None),
        sample_rate_hz=sample_rate,
        metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
    )
