import httpx
import pytest

from app.ai.research import ResearchResult
from app.ai.universal_qa import UniversalQuestionAnswerPipeline


@pytest.mark.asyncio
async def test_universal_pipeline_does_not_need_question_specific_examples() -> None:
    calls: list[str] = []

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            calls.append(user_prompt)
            return "Generated from the user's actual request."

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider())

    reply = await pipeline.answer("Describe a concept that has never appeared in the test suite.")

    assert reply == "Generated from the user's actual request."
    assert len(calls) == 1
    assert "never appeared in the test suite" in calls[0]


@pytest.mark.asyncio
async def test_universal_pipeline_collects_live_research_before_answer() -> None:
    class FakeResearch:
        async def search(self, query: str, limit: int = 5) -> list[ResearchResult]:
            assert query == "latest information about a topic"
            assert limit == 8
            return [
                ResearchResult(
                    "Fresh source",
                    "https://example.com/fresh",
                    "Fresh evidence for the requested topic.",
                ),
                ResearchResult(
                    "Second source",
                    "https://example.org/fresh",
                    "Independent evidence for the requested topic.",
                ),
            ]

    class FakeProvider:
        async def generate(self, *, system_instruction: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
            assert "LIVE RESEARCH REFERENCE:" in user_prompt
            assert "Fresh evidence for the requested topic." in user_prompt
            return "The answer uses the supplied current evidence."

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider(), research_provider=FakeResearch())

    reply = await pipeline.answer("latest information about a topic")

    assert "The answer uses the supplied current evidence." in reply
    assert "Sources:" in reply
    assert "https://example.com/fresh" in reply


@pytest.mark.asyncio
async def test_universal_pipeline_fails_safely_when_current_research_unavailable() -> None:
    class BrokenResearch:
        async def search(self, query: str, limit: int = 5):
            raise RuntimeError("unavailable")

    class FakeProvider:
        async def generate(self, **kwargs):
            raise AssertionError("current answer must not be invented")

    pipeline = UniversalQuestionAnswerPipeline(FakeProvider(), research_provider=BrokenResearch())

    with pytest.raises(RuntimeError, match="live research is unavailable"):
        await pipeline.answer("latest information about a topic")
