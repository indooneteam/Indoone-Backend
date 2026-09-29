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
    calls: list[tuple[str, str, float, int]] = []

    class FakeRuntime:
        def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, language: str) -> str:
            calls.append((prompt, language, temperature, max_new_tokens))
            if len(calls) == 1:
                return "ನೃತ್ಯ ಕಲಿಯುವ ರೋಬೋಟ್‌ನ ಕಥೆ."
            return "The robot practiced dancing every evening and finally performed a funny dance for its friends."

    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_general_knowledge_provider", None)
    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())

    reply = await service.generate_reply("Write a short funny story about a robot learning to dance.")

    assert reply.startswith("The robot practiced dancing")
    assert len(calls) == 2
    prompt, language, temperature, max_new_tokens = calls[0]
    assert prompt.startswith("<instruction>\n")
    assert prompt.endswith("</instruction>\n<response>\n")
    assert "Write a short funny story about a robot learning to dance." in prompt
    assert language == "English"
    assert temperature == 0.7
    assert max_new_tokens == 160
    assert calls[1][2] == 0.2



@pytest.mark.asyncio
async def test_local_model_retries_after_internal_quality_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    class FakeRuntime:
        def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, language: str) -> str:
            calls.append(temperature)
            if len(calls) == 1:
                return "traceback: internal server error"
            return "The robot learned a silly dance and made everyone laugh."

    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_research_provider", None)
    monkeypatch.setattr(service, "_general_knowledge_provider", None)
    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())

    reply = await service.generate_reply("Write a funny story about a robot.")

    assert reply == "The robot learned a silly dance and made everyone laugh."
    assert calls == [0.7, 0.2]


def test_local_knowledge_contains_resilient_india_states_fact() -> None:
    from pathlib import Path

    knowledge_base = LocalKnowledgeBase.from_directory(Path("data/knowledge"))
    hits = knowledge_base.search("How many states are there in India?", limit=3)

    answer = _knowledge_fallback_sentence(
        "How many states are there in India?",
        hits,
    )

    assert answer == "India has 28 states and 8 Union Territories."


@pytest.mark.asyncio
async def test_research_generation_uses_full_evidence_context_for_one_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResearchProvider:
        async def search(self, query: str, limit: int = 5):
            assert query == "ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ"
            assert limit == 8
            return [
                service.ResearchResult(
                    "AI research source",
                    "https://research.example/ai",
                    "Generative AI is an active research area.",
                ),
                service.ResearchResult(
                    "AI news source",
                    "https://news.example/ai",
                    "Recent AI developments include multimodal systems.",
                ),
            ]

    calls: list[str] = []

    class FakeRuntime:
        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            language: str,
        ) -> str:
            calls.append(prompt)
            return "AI technology ಹಲವು research areasನಲ್ಲಿ ಅಭಿವೃದ್ಧಿಯಾಗುತ್ತಿದೆ."

    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_general_knowledge_provider", None)
    monkeypatch.setattr(service, "_research_provider", FakeResearchProvider())
    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())

    reply = await service.generate_reply("ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ")

    assert len(calls) == 1
    assert "Fresh research evidence:" in calls[0]
    assert "research.example" in calls[0]
    assert "news.example" in calls[0]
    assert "Sources:" in reply
    assert reply.count("https://research.example/ai") == 1
    assert reply.count("https://news.example/ai") == 1


@pytest.mark.asyncio
async def test_research_generation_failure_does_not_attach_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResearchProvider:
        async def search(self, query: str, limit: int = 5):
            return [
                service.ResearchResult(
                    "Relevant AI source",
                    "https://research.example/ai",
                    "AI technology evidence.",
                ),
                service.ResearchResult(
                    "Second AI source",
                    "https://second.example/ai",
                    "Additional AI technology evidence.",
                ),
            ]

    class FakeRuntime:
        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            language: str,
        ) -> str:
            return "This is not Kannada."

    monkeypatch.setattr(service, "_knowledge_base", None)
    monkeypatch.setattr(service, "_general_knowledge_provider", None)
    monkeypatch.setattr(service, "_research_provider", FakeResearchProvider())
    monkeypatch.setattr(service, "_load_local_model_runtime", lambda: FakeRuntime())

    reply = await service.generate_reply("ಈಗಿನ AI technology ಬಗ್ಗೆ research ಮಾಡಿ")

    assert "Sources:" not in reply
    assert "research.example" not in reply
    assert "reliable information" in reply.lower()


def test_research_requires_independent_sources() -> None:
    results = [
        service.ResearchResult("One", "https://example.com/one", "source"),
        service.ResearchResult("Two", "https://example.com/two", "source"),
    ]
    assert service._research_has_enough_sources(results, cross_check=False) is False

    results.append(
        service.ResearchResult("Three", "https://second.example/three", "source")
    )
    assert service._research_has_enough_sources(results, cross_check=False) is True
    assert service._research_has_enough_sources(results, cross_check=True) is False

