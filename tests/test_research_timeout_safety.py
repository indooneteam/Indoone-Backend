import pytest

from app.ai.research import HttpResearchProvider


def test_research_provider_rejects_excessive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must not exceed 60 seconds"):
        HttpResearchProvider("https://search.example/api", timeout=60.1)
