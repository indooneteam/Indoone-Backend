import pytest

from app.ai.sarvam_runtime import _render_prompt
from app.ai.sarvam_service import generate_sarvam_reply


def test_render_prompt_uses_verified_sarvam_inst_format() -> None:
    prompt = _render_prompt(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert prompt == "[INST] <<SYS>>\nBe helpful.\n<</SYS>>\nHello [/INST]"
    assert prompt.startswith("[INST]")
    assert prompt.endswith("[/INST]")


def test_render_prompt_keeps_multi_turn_history() -> None:
    prompt = _render_prompt(
        [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
    )

    assert "[INST] <<SYS>>\nBe helpful." in prompt
    assert "Hello! </s>[INST] How are you?" in prompt


@pytest.mark.asyncio
async def test_generate_sarvam_reply_uses_candidate_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntime:
        def generate(self, messages, **kwargs) -> str:
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "Namaskara"
            return "Namaskara!"

    monkeypatch.setattr("app.ai.sarvam_service._load_runtime", lambda: FakeRuntime())

    reply = await generate_sarvam_reply("Namaskara")

    assert reply == "Namaskara!"

@pytest.mark.asyncio
async def test_generate_sarvam_reply_uses_compact_web_context(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRuntime:
        def generate(self, messages, **kwargs) -> str:
            content = messages[-1]["content"]
            assert "VERIFIED WEB EVIDENCE:" in content
            assert "LIVE WEB EVIDENCE:" not in content
            assert "USER-PROVIDED DOCUMENT:" not in content
            assert "Do not reproduce the evidence block" in content
            return "Bengaluru is the capital of Karnataka."

    monkeypatch.setattr("app.ai.sarvam_service._load_runtime", lambda: FakeRuntime())

    reply = await generate_sarvam_reply(
        "Karnataka da rajadhani yavudu?",
        document_context=(
            "LIVE WEB EVIDENCE:\n"
            "<research>\n"
            "result: 1\n"
            "source: karnataka.gov.in\n"
            "title: Karnataka\n"
            "evidence: Bengaluru is the capital of Karnataka.\n"
            "</research>"
        ),
    )

    assert reply == "Bengaluru is the capital of Karnataka."
