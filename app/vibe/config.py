from __future__ import annotations

from dataclasses import dataclass
import os


DEFAULT_MODEL = "gemini-3.8-live-extended-thinking"
DEFAULT_VOICE = "Puck"
DEFAULT_THINKING_LEVEL = "low"


@dataclass(frozen=True)
class VibeConfig:
    api_key: str
    model: str
    voice: str
    thinking_level: str
    endpoint: str
    system_instruction: str
    input_transcription: bool
    output_transcription: bool

    @classmethod
    def from_environment(cls) -> "VibeConfig":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise RuntimeError("GEMINI_LIVE_MODEL is not configured")

        voice = os.getenv("GEMINI_LIVE_VOICE", DEFAULT_VOICE).strip()
        if not voice:
            raise RuntimeError("GEMINI_LIVE_VOICE is not configured")

        thinking_level = os.getenv(
            "GEMINI_LIVE_THINKING_LEVEL",
            DEFAULT_THINKING_LEVEL,
        ).strip().casefold()
        if thinking_level not in {"low", "medium", "high"}:
            raise RuntimeError(
                "GEMINI_LIVE_THINKING_LEVEL must be low, medium, or high"
            )

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
            thinking_level=thinking_level,
            endpoint=endpoint,
            system_instruction=(
                "You are Indoone Vibe, the real-time voice assistant inside "
                "the Indoone app. Respond naturally and conversationally. "
                "Answer in the user's language. Understand Romanized Kannada "
                "as Kannada. Keep spoken responses clear and reasonably concise. "
                "Never reveal internal provider, model, API key, or backend details."
            ),
            input_transcription=True,
            output_transcription=True,
        )

    def websocket_url(self) -> str:
        separator = "&" if "?" in self.endpoint else "?"
        return f"{self.endpoint}{separator}key={self.api_key}"

    def setup_message(self) -> dict[str, object]:
        generation_config: dict[str, object] = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": self.voice,
                    }
                }
            },
            "thinkingConfig": {
                "thinkingLevel": self.thinking_level.upper(),
            },
        }

        setup: dict[str, object] = {
            "model": f"models/{self.model}",
            "generationConfig": generation_config,
            "systemInstruction": {
                "parts": [{"text": self.system_instruction}],
            },
        }

        if self.input_transcription:
            setup["inputAudioTranscription"] = {}
        if self.output_transcription:
            setup["outputAudioTranscription"] = {}

        return {"setup": setup}
