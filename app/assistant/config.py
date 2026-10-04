from __future__ import annotations

from dataclasses import dataclass
import os

DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Puck"


@dataclass(frozen=True)
class AssistantConfig:
    api_key: str
    model: str
    voice: str
    endpoint: str
    system_instruction: str
    input_transcription: bool
    output_transcription: bool

    @classmethod
    def from_environment(cls) -> "AssistantConfig":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        model = os.getenv("GEMINI_ASSISTANT_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise RuntimeError("GEMINI_ASSISTANT_MODEL is not configured")

        voice = os.getenv("GEMINI_ASSISTANT_VOICE", DEFAULT_VOICE).strip()
        if not voice:
            raise RuntimeError("GEMINI_ASSISTANT_VOICE is not configured")

        endpoint = os.getenv(
            "GEMINI_LIVE_WS_URL",
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
        ).strip()
        if not endpoint.startswith("wss://"):
            raise RuntimeError("GEMINI_LIVE_WS_URL must use wss://")

        return cls(
            api_key=api_key,
            model=model.removeprefix("models/"),
            voice=voice,
            endpoint=endpoint,
            system_instruction=(
                "You are Indoone Home Assistant, a fast real-time voice assistant "
                "for the Indoone home experience. Respond naturally in the user's "
                "language. Understand Romanized Kannada as Kannada. Keep spoken "
                "answers useful, direct, and concise. Do not mention internal "
                "provider, model, API key, or backend details."
            ),
            input_transcription=True,
            output_transcription=True,
        )

    def websocket_url(self) -> str:
        separator = "&" if "?" in self.endpoint else "?"
        return f"{self.endpoint}{separator}key={self.api_key}"

    def setup_message(self) -> dict[str, object]:
        setup: dict[str, object] = {
            "model": f"models/{self.model}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.voice,
                        }
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": self.system_instruction}],
            },
        }

        if self.input_transcription:
            setup["inputAudioTranscription"] = {}
        if self.output_transcription:
            setup["outputAudioTranscription"] = {}

        return {"setup": setup}
