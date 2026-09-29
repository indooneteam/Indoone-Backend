import pytest

from app.ai import service
from app.ai.general_knowledge import WikipediaAnswer
from app.ai.knowledge import LocalKnowledgeBase
from app.ai.service import _knowledge_fallback_sentence


def test_knowledge_fallback_returns_matching_sentence(tmp_path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "facts.txt").write_text(
        "The capital city of India is New Delhi.\n"
        "Day and night happen because Earth rotates on its axis.\n",
        encoding="utf-8",
    )

    knowledge_base = LocalKnowledgeBase.from_directory(knowledge_dir)
    hits = knowledge_base.search("What is the capital city of India?", limit=3)

    answer = _knowledge_fallback_sentence(
        "What is the capital city of India?",
        hits,
    )

    assert answer == "The capital city of India is New Delhi."


def test_knowledge_fallback_preserves_matching_script(tmp_path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "facts.txt").write_text(
        "The capital city of India is New Delhi.\n"
        "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ.\n",
        encoding="utf-8",
    )

    knowledge_base = LocalKnowledgeBase.from_directory(knowledge_dir)
    hits = knowledge_base.search("ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?", limit=3)

    answer = _knowledge_fallback_sentence(
        "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು?",
        hits,
    )

    assert answer == "ಭಾರತದ ರಾಜಧಾನಿ ನವದೆಹಲಿ."


@pytest.mark.asyncio
async def test_general_knowledge_answer_bypasses_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProvider:
        async def answer(self, query: str, language: str = "English") -> WikipediaAnswer | None:
            assert query == "What is photosynthesis?"
            assert language == "English"
            return WikipediaAnswer(
                title="Photosynthesis",
                url="https://en.wikipedia.org/wiki/Photosynthesis",
                extract="Photosynthesis is the process by which green plants convert light energy into chemical energy.",
            )

    async def fail_model(*args, **kwargs):
        raise AssertionError("local model must not run for an ordinary factual question")

    monkeypatch.setattr(service, "_general_knowledge_provider", FakeProvider())
    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_load_local_model_runtime", fail_model)

    reply = await service.generate_reply("What is photosynthesis?")

    assert reply.startswith("Photosynthesis is the process")
    assert "Sources:" in reply
    assert "https://en.wikipedia.org/wiki/Photosynthesis" in reply


@pytest.mark.asyncio
async def test_local_model_uses_training_prompt_contract_and_language_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeRuntime:
        def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, language: str) -> str:
            calls.append((prompt, language))
            if len(calls) == 1:
                return "ನೃತ್ಯ ಕಲಿಯುವ ರೋಬೋಟ್‌ನ ಕಥೆ."
            return "The robot practiced dancing every evening and finally performed a funny dance for its friends."

    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_general_knowledge_provider", None)
    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())

    reply = await service.generate_reply("Write a short funny story about a robot learning to dance.")

    assert reply.startswith("The robot practiced dancing")
    assert len(calls) == 1
    prompt, language = calls[0]
    assert prompt.startswith("<instruction>\n")
    assert prompt.endswith("</instruction>\n<response>\n")
    assert "Write a short funny story about a robot learning to dance." in prompt
    assert language == "English"

