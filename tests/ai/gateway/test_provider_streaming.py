"""The provider stream generators, against mock SSE frames.

These exercise the shape of the two providers' streaming responses - which is what a unit
test can honestly check. They do not prove the live endpoints behave this way; that is a
deployment check, listed in the PR.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.ai.gateway import azure_openai_service, gemini_service, service


class _FakeResponse:
    def __init__(self, lines, status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b'{"error": "nope"}'


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *_exc) -> bool:
        return False


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_exc) -> bool:
        return False

    def stream(self, *_args, **_kwargs) -> _FakeStream:
        return _FakeStream(self._response)


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def _collect(agen) -> list[str]:
    async def run():
        return [chunk async for chunk in agen]

    return asyncio.run(run())


# ---- Gemini ---------------------------------------------------------------


def test_gemini_yields_each_text_delta(monkeypatch) -> None:
    lines = [
        _sse({"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}),
        "",
        _sse({"candidates": [{"content": {"parts": [{"text": "lo"}]}}]}),
        _sse({"candidates": [{"content": {"parts": [{"text": " there"}]}}], "usageMetadata": {"totalTokenCount": 3}}),
    ]
    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("key", "gemini-test"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", lambda **_k: _FakeClient(_FakeResponse(lines)))

    chunks = _collect(gemini_service.stream_gemini_reply("hi"))

    assert chunks == ["Hel", "lo", " there"]


def test_gemini_skips_frames_without_text(monkeypatch) -> None:
    lines = [
        _sse({"usageMetadata": {"totalTokenCount": 2}}),
        _sse({"candidates": [{"finishReason": "STOP"}]}),
        _sse({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}),
    ]
    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("key", "gemini-test"))
    monkeypatch.setattr(gemini_service.httpx, "AsyncClient", lambda **_k: _FakeClient(_FakeResponse(lines)))

    assert _collect(gemini_service.stream_gemini_reply("hi")) == ["ok"]


def test_gemini_reports_a_failed_stream(monkeypatch) -> None:
    monkeypatch.setattr(gemini_service, "_get_config", lambda: ("key", "gemini-test"))
    monkeypatch.setattr(
        gemini_service.httpx, "AsyncClient",
        lambda **_k: _FakeClient(_FakeResponse([], status_code=500)),
    )

    with pytest.raises(RuntimeError):
        _collect(gemini_service.stream_gemini_reply("hi"))


# ---- Azure ----------------------------------------------------------------


def test_azure_yields_deltas_and_ignores_the_done_frame(monkeypatch) -> None:
    lines = [
        _sse({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse({"choices": [{"delta": {"content": " there"}}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        azure_openai_service, "_get_config",
        lambda: ("key", "https://example.openai.azure.com", "deployment", "2024-08-01"),
    )
    monkeypatch.setattr(
        azure_openai_service.httpx, "AsyncClient",
        lambda **_k: _FakeClient(_FakeResponse(lines)),
    )

    chunks = _collect(azure_openai_service.stream_azure_reply("hi"))

    assert chunks == ["Hello", " there"]


# ---- the gateway ----------------------------------------------------------


def test_a_stream_that_fails_before_the_first_chunk_falls_back(monkeypatch) -> None:
    async def broken_stream(*_args, **_kwargs):
        raise RuntimeError("provider refused")
        yield ""  # pragma: no cover - makes this an async generator

    async def blocking_reply(*_args, **_kwargs):
        return "the whole answer"

    monkeypatch.delenv("INDOONE_STREAM_PROVIDERS", raising=False)
    monkeypatch.setattr(service, "_provider_stream", lambda: broken_stream)
    monkeypatch.setattr(service, "generate_reply", blocking_reply)

    assert _collect(service.stream_generate("hi")) == ["the whole answer"]


def test_a_stream_that_fails_after_the_first_chunk_propagates(monkeypatch) -> None:
    async def half_broken(*_args, **_kwargs):
        yield "partial"
        raise RuntimeError("connection dropped")

    monkeypatch.delenv("INDOONE_STREAM_PROVIDERS", raising=False)
    monkeypatch.setattr(service, "_provider_stream", lambda: half_broken)

    with pytest.raises(RuntimeError):
        _collect(service.stream_generate("hi"))
