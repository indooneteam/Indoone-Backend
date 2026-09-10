import os

from openai import AsyncOpenAI


SYSTEM_PROMPT = """You are Indoone AI, the assistant inside the Indoone app.
Be helpful, clear, concise, and natural. Do not claim to have accessed company
systems or user data unless a tool actually provides that data."""


async def generate_reply(message: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = AsyncOpenAI(api_key=api_key)

    response = await client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=message,
    )
    return response.output_text
