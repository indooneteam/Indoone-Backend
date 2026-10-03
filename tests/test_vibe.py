from __future__ import annotations

from app.vibe.config import VibeConfig
from app.vibe.gemini_live import translate_upstream_event


def test_vibe_setup_message_uses_selected_live_model() -> None:
    config = VibeConfig(
        api_key="test-key",
        model="gemini-3.8-live-extended-thinking",
        voice="Puck",
        thinking_level="low",
        endpoint="wss://example.test/live",
        system_instruction="Indoone Vibe",
        input_transcription=True,
        output_transcription=True,
    )

    setup = config.setup_message()["setup"]
    assert setup["model"] == "models/gemini-3.8-live-extended-thinking"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "LOW"
    assert "inputAudioTranscription" in setup
    assert "outputAudioTranscription" in setup


def test_vibe_translates_audio_and_transcripts() -> None:
    payload = {
        "serverContent": {
            "interactionStatus": "IN_PROGRESS",
            "inputTranscription": {"text": "Hello"},
            "outputTranscription": {"text": "Hi there"},
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": "AQID",
                        }
                    }
                ]
            },
            "turnComplete": True,
        }
    }

    events = translate_upstream_event(payload)

    assert {"type": "status", "interaction_status": "IN_PROGRESS"} in events
    assert {"type": "transcript", "role": "user", "final": True, "text": "Hello"} in events
    assert {"type": "transcript", "role": "assistant", "final": True, "text": "Hi there"} in events
    assert {"type": "audio", "audio_base64": "AQID", "mime_type": "audio/pcm;rate=24000"} in events
    assert {"type": "turn_complete", "interaction_status": "IN_PROGRESS"} in events
