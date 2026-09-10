"""Indoone AI core entry point.

No hosted AI provider is used here. The model runtime is isolated behind a
local engine so Indoone can evolve toward its own trained model/checkpoint
without changing the public chat API.
"""

from app.ai.local_engine import LocalAIEngine


_engine = LocalAIEngine()


async def generate_reply(message: str) -> str:
    return await _engine.generate(message)
