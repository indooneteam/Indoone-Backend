"""Local Indoone AI engine boundary.

The first implementation is intentionally small and deterministic. It gives
us a real local execution path today while keeping the model runtime isolated
so a trained Indoone checkpoint can replace it later without changing the API.
"""


class LocalAIEngine:
    async def generate(self, message: str) -> str:
        text = message.strip()
        lowered = text.lower()

        if lowered in {"hi", "hello", "hey", "namaste"}:
            return "Hello! I'm Indoone AI. How can I help you?"

        if lowered.endswith("?"):
            return (
                "I understand your question. Indoone AI's local model runtime "
                "is being built now, and this chat path is ready for the model "
                "to be plugged in next."
            )

        return f"I received your message: {text}"
