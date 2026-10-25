"""Phase 0 observability, pinned by tests.

Each test covers one promise the module makes: a span is measured and timed, a
failure is marked without inventing one, tokens the provider reports are counted, cost
is computed only when a price is configured, a tool call and a request each produce a
span, and the operator endpoint is off until it is turned on.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.ai.gateway import observability
from app.ai.execution import tools as ai_tools
from app.ai.gateway.gemini_service import _record_usage as record_gemini_usage
from app.ai.gateway.azure_openai_service import _record_usage as record_azure_usage
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from a clean window and no configured prices."""

    monkeypatch.delenv("INDOONE_MODEL_PRICES_JSON", raising=False)
    monkeypatch.delenv("INDOONE_OBSERVABILITY_TOKEN", raising=False)
    observability.reset()


def test_a_span_is_timed_and_recorded_on_the_request_trace() -> None:
    observability.begin_request("req-1")

    with observability.track("work", "tool", tool="text_stats") as span:
        span.attributes["ok"] = True

    trace = observability.current_trace()
    assert trace is not None
    assert [recorded.name for recorded in trace.spans] == ["work"]
    assert trace.spans[0].duration_ms >= 0
    assert trace.spans[0].status == "ok"
    assert trace.spans[0].attributes == {"tool": "text_stats", "ok": True}
    assert observability.snapshot()["spans"]["total"]["tool"] == 1


def test_a_raised_exception_marks_the_span_failed_without_hiding_it() -> None:
    with pytest.raises(RuntimeError):
        with observability.track("work", "model", model="gemini"):
            raise RuntimeError("provider exploded")

    snapshot = observability.snapshot()
    assert snapshot["spans"]["total"]["model"] == 1
    assert snapshot["spans"]["errors"]["model"] == 1
    assert snapshot["recent_spans"][0]["error"] == "RuntimeError"


def test_a_block_that_reports_failure_can_say_so() -> None:
    with observability.track("work", "tool", tool="github_send") as span:
        span.fail("tool_error")

    snapshot = observability.snapshot()
    assert snapshot["spans"]["errors"]["tool"] == 1
    assert snapshot["tools"]["github_send"]["errors"] == 1


def test_tool_stats_are_grouped_by_tool_name() -> None:
    with observability.track("tool.call", "tool", tool="text_stats") as first:
        first.attributes["ok"] = True
    with observability.track("tool.call", "tool", tool="github_list_repositories") as second:
        second.attributes["ok"] = False
        second.fail("tool_error")

    tools = observability.snapshot()["tools"]
    assert set(tools) == {"text_stats", "github_list_repositories"}
    assert tools["text_stats"]["calls"] == 1
    assert tools["github_list_repositories"]["errors"] == 1


def test_provider_reported_tokens_are_counted_and_accumulate() -> None:
    @observability.measured_model_call("gemini")
    async def call() -> str:
        observability.set_model("gemma-test")
        record_gemini_usage({"usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200}})
        record_gemini_usage({"usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10}})
        return "ok"

    assert asyncio.run(call()) == "ok"
    models = observability.snapshot()["models"]
    assert models["gemma-test"]["calls"] == 1
    assert models["gemma-test"]["input_tokens"] == 1050
    assert models["gemma-test"]["output_tokens"] == 210


def test_azure_usage_is_read_from_its_own_shape() -> None:
    @observability.measured_model_call("azure")
    async def call() -> str:
        observability.set_model("gpt-test")
        record_azure_usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        return "ok"

    asyncio.run(call())
    models = observability.snapshot()["models"]
    assert models["gpt-test"]["input_tokens"] == 7
    assert models["gpt-test"]["output_tokens"] == 3


def test_usage_without_a_payload_is_ignored_rather_than_guessed() -> None:
    @observability.measured_model_call("gemini")
    async def call() -> str:
        observability.set_model("gemma-test")
        record_gemini_usage({"candidates": []})
        record_gemini_usage({"usageMetadata": {"promptTokenCount": "not a number"}})
        return "ok"

    asyncio.run(call())
    models = observability.snapshot()["models"]
    assert models["gemma-test"]["input_tokens"] == 0
    assert models["gemma-test"]["output_tokens"] == 0


def test_cost_is_unpriced_until_a_price_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "INDOONE_MODEL_PRICES_JSON",
        json.dumps({"gemma-test": {"input": 1.0, "output": 2.0}}),
    )

    @observability.measured_model_call("gemini")
    async def call() -> str:
        observability.set_model("gemma-test")
        record_gemini_usage({"usageMetadata": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 1_000_000}})
        return "ok"

    asyncio.run(call())
    stats = observability.snapshot()["models"]["gemma-test"]
    assert stats["cost_usd"] == pytest.approx(3.0)
    assert stats["unpriced_calls"] == 0


def test_an_unpriced_model_is_reported_as_unpriced_not_as_free() -> None:
    @observability.measured_model_call("gemini")
    async def call() -> str:
        observability.set_model("gemma-test")
        record_gemini_usage({"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10}})
        return "ok"

    asyncio.run(call())
    stats = observability.snapshot()["models"]["gemma-test"]
    assert stats["cost_usd"] == 0
    assert stats["unpriced_calls"] == 1


def test_a_real_tool_call_produces_a_tool_span() -> None:
    result = asyncio.run(ai_tools.run_tool_async("text_stats", json.dumps({"text": "hello there"})))

    assert result.safe is True
    tools = observability.snapshot()["tools"]
    assert tools["text_stats"]["calls"] == 1
    assert tools["text_stats"]["errors"] == 0


def test_a_failing_tool_call_is_marked_failed() -> None:
    result = asyncio.run(ai_tools.run_tool_async("calculator", "not json"))

    assert result.safe is False
    tools = observability.snapshot()["tools"]
    assert tools["calculator"]["calls"] == 1
    assert tools["calculator"]["errors"] == 1


def test_a_provider_fallback_is_counted() -> None:
    from app.ai.gateway.gemini_service import ModelBusyError, run_with_provider_fallback

    async def primary() -> str:
        raise ModelBusyError("busy")

    async def fallback() -> str:
        return "the other provider answered"

    assert asyncio.run(run_with_provider_fallback(primary, fallback)) == "the other provider answered"
    assert observability.snapshot()["provider_fallbacks"] == 1


def test_a_request_produces_a_request_span() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    recent = observability.snapshot()["recent_spans"]
    request_spans = [span for span in recent if span["kind"] == "request" and span["name"] == "GET /health"]
    assert request_spans
    assert request_spans[-1]["attributes"]["status_code"] == 200


def test_the_snapshot_carries_no_prompt_payload_or_user_id() -> None:
    secret = "meera@example.com please send the invoice"
    asyncio.run(ai_tools.run_tool_async("text_stats", json.dumps({"text": secret})))

    rendered = json.dumps(observability.snapshot())
    assert secret not in rendered
    assert "meera@example.com" not in rendered
    assert "no prompts, payloads or user ids" in rendered


def test_the_endpoint_is_off_until_an_operator_turns_it_on() -> None:
    response = client.get("/api/observability")

    assert response.status_code == 404


def test_the_endpoint_refuses_the_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_OBSERVABILITY_TOKEN", "operator-secret")

    response = client.get("/api/observability", headers={"X-Indoone-Observability-Token": "guess"})

    assert response.status_code == 401


def test_the_endpoint_returns_the_snapshot_for_the_operator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_OBSERVABILITY_TOKEN", "operator-secret")
    with observability.track("work", "tool", tool="text_stats"):
        pass

    response = client.get("/api/observability", headers={"X-Indoone-Observability-Token": "operator-secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["tools"]["text_stats"]["calls"] == 1
    assert "note" in body


def test_the_operator_can_reset_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDOONE_OBSERVABILITY_TOKEN", "operator-secret")
    with observability.track("work", "tool", tool="text_stats"):
        pass

    response = client.post(
        "/api/observability/reset",
        headers={"X-Indoone-Observability-Token": "operator-secret"},
    )

    assert response.status_code == 200
    assert observability.snapshot()["tools"] == {}
