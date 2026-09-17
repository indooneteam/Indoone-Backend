from __future__ import annotations

import asyncio
import base64

import app.capabilities.voice_session as voice_session


def test_voice_session_cancellation_state_and_ready_event() -> None:
    state = voice_session.VoiceSessionState()
    ready = voice_session.ready_event(state)

    assert ready["type"] == "ready"
    assert "cancellation" in ready["capabilities"]
    assert state.cancel_request("req-1") is True
    assert state.is_cancelled("req-1") is True
    assert state.is_cancelled("req-2") is False


def test_cancelled_stt_request_returns_cancelled_event(monkeypatch) -> None:
    async def fake_transcribe(*args, **kwargs):
        return voice_session.TranscriptionResult(
            text="namaskara",
            language="kn",
            provider="local",
            model="stt-test",
            confidence=0.9,
            metadata={},
        )

    monkeypatch.setattr(voice_session, "transcribe_audio", fake_transcribe)
    state = voice_session.VoiceSessionState()
    state.cancel_request("req-stt")

    event = asyncio.run(
        voice_session.handle_audio_message(
            state,
            base64.b64encode(b"fake-audio").decode("ascii"),
            request_id="req-stt",
        )
    )

    assert event["type"] == "cancelled"
    assert event["request_id"] == "req-stt"


def test_cancelled_tts_request_returns_cancelled_event(monkeypatch) -> None:
    async def fake_synthesize(*args, **kwargs):
        return voice_session.SynthesisResult(
            audio_base64=base64.b64encode(b"fake-wav").decode("ascii"),
            mime_type="audio/wav",
            provider="local",
            model="tts-test",
            sample_rate_hz=16_000,
            metadata={},
        )

    monkeypatch.setattr(voice_session, "synthesize_speech", fake_synthesize)
    state = voice_session.VoiceSessionState()
    state.cancel_request("req-tts")

    event = asyncio.run(voice_session.handle_speak_message(state, "namaskara", request_id="req-tts"))

    assert event["type"] == "cancelled"
    assert event["request_id"] == "req-tts"
