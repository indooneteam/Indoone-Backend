import pytest

from app.ai.qwen_service import _strip_think_content, generate_qwen_reply


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "<think>hidden reasoning</think>\nHello! How can I assist you today?",
            "Hello! How can I assist you today?",
        ),
        (
            "Before\n<think>hidden</think>\nAfter",
            "Before\n\nAfter",
        ),
        (
            "<THINK>hidden reasoning</THINK>\nFinal answer",
            "Final answer",
        ),
        (
            "Final answer\n<think>unfinished reasoning",
            "Final answer",
        ),
    ],
)
def test_strip_think_content(raw: str, expected: str) -> None:
    assert _strip_think_content(raw) == expected


@pytest.mark.asyncio
async def test_generate_qwen_reply_strips_think_content(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntime:
        def generate(self, *args, **kwargs) -> str:
            return "<think>internal reasoning</think>\nHello!"

    monkeypatch.setattr("app.ai.qwen_service._load_runtime", lambda: FakeRuntime())

    reply = await generate_qwen_reply("hi")

    assert reply == "Hello!"
    assert "<think>" not in reply.lower()
    assert "</think>" not in reply.lower()
