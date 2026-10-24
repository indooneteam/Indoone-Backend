"""One bridge, two providers: the fallback must fire, and only when it should.

These tests never reach the network. They pin the behaviour the whole design rests
on: the primary provider answers normally, and the other provider answers only when
the primary reports that it is at its limit.
"""

import asyncio
import importlib

import pytest


@pytest.fixture()
def bridge():
    return importlib.import_module("app.ai.gemini_service")


def test_the_primary_answer_is_returned(bridge):
    async def primary():
        return "from the primary"

    async def fallback():
        return "from the fallback"

    assert asyncio.run(bridge.run_with_provider_fallback(primary, fallback)) == "from the primary"


def test_the_fallback_fires_only_when_the_primary_is_busy(bridge):
    calls = {"fallback": 0}

    async def primary():
        raise bridge.ModelBusyError(bridge.BUSY_MESSAGE)

    async def fallback():
        calls["fallback"] += 1
        return "from the fallback"

    assert asyncio.run(bridge.run_with_provider_fallback(primary, fallback)) == "from the fallback"
    assert calls["fallback"] == 1


def test_a_real_error_is_not_swallowed(bridge):
    """Only "busy" falls back. Any other failure must still surface."""

    async def primary():
        raise RuntimeError("a genuine failure")

    async def fallback():
        return "from the fallback"

    with pytest.raises(RuntimeError):
        asyncio.run(bridge.run_with_provider_fallback(primary, fallback))


def test_the_transport_speaks_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    azure = importlib.import_module("app.ai.azure_openai_service")
    url = azure._chat_url("https://example.openai.azure.com", "gpt-4o", "2024-10-21")
    assert url.startswith("https://example.openai.azure.com/openai/deployments/gpt-4o/chat/completions")
    assert "api-version=" in url
