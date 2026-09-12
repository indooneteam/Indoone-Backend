from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from app.capabilities.voice import SynthesisResult, TranscriptionResult, synthesize_speech, transcribe_audio

MAX_SESSION_MESSAGES = 200
MAX_REQUEST_ID_LENGTH = 128


@dataclass
class VoiceSessionState:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    sequence: int = 0
    messages: int = 0
    closed: bool = False

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def accept_message(self) -> None:
        self.messages += 1
        if self.messages > MAX_SESSION_MESSAGES:
            raise ValueError("voice session message limit reached")


def normalize_request_id(value: object) -> str | None:
    if value is None:
        return None
    request_id = str(value).strip()
    return request_id[:MAX_REQUEST_ID_LENGTH] or None


def ready_event(state: VoiceSessionState) -> dict[str, object]:
    return {
        "type": "ready",
        "protocol": "indoone.voice.v1",
        "session_id": state.session_id,
        "capabilities": ["stt", "tts", "partial_transcripts", "request_ids"],
        "sequence": state.next_sequence(),
    }


def transcript_event(
    state: VoiceSessionState,
    result: TranscriptionResult,
    *,
    final: bool,
    request_id: str | None = None,
) -> dict[str, object]:
    return {
        "type": "transcript",
        "final": final,
        "request_id": request_id,
        "text": result.text,
        "language": result.language,
        "provider": result.provider,
        "model": result.model,
        "confidence": result.confidence,
        "sequence": state.next_sequence(),
    }


def audio_event(
    state: VoiceSessionState,
    result: SynthesisResult,
    *,
    request_id: str | None = None,
) -> dict[str, object]:
    return {
        "type": "audio",
        "request_id": request_id,
        "audio_base64": result.audio_base64,
        "mime_type": result.mime_type,
        "provider": result.provider,
        "model": result.model,
        "sample_rate_hz": result.sample_rate_hz,
        "sequence": state.next_sequence(),
    }


async def handle_audio_message(
    state: VoiceSessionState,
    audio_base64: str,
    *,
    mime_type: str = "audio/wav",
    language: str = "",
    final: bool = True,
    request_id: str | None = None,
) -> dict[str, object]:
    state.accept_message()
    result = await transcribe_audio(audio_base64, mime_type=mime_type, language=language)
    return transcript_event(state, result, final=final, request_id=request_id)


async def handle_speak_message(
    state: VoiceSessionState,
    text: str,
    *,
    language: str = "",
    voice: str = "",
    format: str = "wav",
    request_id: str | None = None,
) -> dict[str, object]:
    state.accept_message()
    result = await synthesize_speech(text, language=language, voice=voice, format=format)
    return audio_event(state, result, request_id=request_id)
