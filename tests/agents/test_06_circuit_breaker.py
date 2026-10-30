"""Agent 06 - the Circuit Breaker.

The property under test is a policy, not a code path: **Gemini answers unless Gemini is
unwell.** Everything else here follows from that - three failures open the circuit, an open
circuit sends traffic straight to Azure with no wait, and a canary closes it again.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agents import circuit_breaker as breaker


@pytest.fixture(autouse=True)
def _fresh():
    breaker.reset()
    yield
    breaker.reset()


# ---------------------------------------------------------------- 1. Gemini first


def test_a_healthy_system_sends_every_turn_to_gemini() -> None:
    assert breaker.preferred_order() == (breaker.GEMINI, breaker.AZURE)
    assert breaker.state_of(breaker.GEMINI) == breaker.CLOSED


def test_gemini_first_is_a_policy_not_a_configuration() -> None:
    """The order does not depend on which backend a deployment selected."""
    assert breaker.preferred_order()[0] == breaker.GEMINI


# ---------------------------------------------------------------- 2. tripping


def test_three_consecutive_failures_open_the_circuit() -> None:
    moment = 1000.0

    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=moment + index)
        if index < 2:
            assert breaker.state_of(breaker.GEMINI, now=moment + index) == breaker.CLOSED

    assert breaker.state_of(breaker.GEMINI, now=moment + 2) == breaker.OPEN


def test_two_failures_are_not_an_outage() -> None:
    for index in range(2):
        breaker.note_failure(breaker.GEMINI, "500", now=1000.0 + index)

    assert breaker.state_of(breaker.GEMINI, now=1002.0) == breaker.CLOSED


def test_failures_spread_beyond_the_window_do_not_trip_it() -> None:
    """A provider that fails once a minute is unlucky, not degraded."""
    breaker.note_failure(breaker.GEMINI, "500", now=1000.0)
    breaker.note_failure(breaker.GEMINI, "500", now=1000.0 + 61)
    breaker.note_failure(breaker.GEMINI, "500", now=1000.0 + 122)

    assert breaker.state_of(breaker.GEMINI, now=1000.0 + 122) == breaker.CLOSED


def test_a_success_clears_the_count() -> None:
    breaker.note_failure(breaker.GEMINI, "500", now=1000.0)
    breaker.note_failure(breaker.GEMINI, "500", now=1001.0)
    breaker.note_success(breaker.GEMINI, now=1002.0)
    breaker.note_failure(breaker.GEMINI, "500", now=1003.0)

    assert breaker.state_of(breaker.GEMINI, now=1003.0) == breaker.CLOSED


# ---------------------------------------------------------------- 3. open means Azure


def test_an_open_circuit_sends_traffic_to_azure_immediately() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    order = breaker.preferred_order(now=1003.0)

    assert order == (breaker.AZURE, breaker.GEMINI)
    assert breaker.state_of(breaker.AZURE, now=1003.0) == breaker.CLOSED


def test_the_open_state_costs_nothing_to_check() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    started = time.perf_counter()
    for _ in range(1000):
        breaker.preferred_order(now=1003.0)
    elapsed_ms = (time.perf_counter() - started) * 1000 / 1000

    assert elapsed_ms < 0.05, f"decision cost {elapsed_ms:.4f} ms"


# ---------------------------------------------------------------- 4. recovery


def test_the_cooldown_turns_open_into_half_open() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    assert breaker.state_of(breaker.GEMINI, now=1003.0) == breaker.OPEN
    assert breaker.state_of(breaker.GEMINI, now=1003.0 + breaker.COOLDOWN_SECONDS) == breaker.HALF_OPEN


def test_one_canary_at_a_time() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    later = 1003.0 + breaker.COOLDOWN_SECONDS

    assert breaker.should_probe(breaker.GEMINI, now=later) is True
    assert breaker.should_probe(breaker.GEMINI, now=later) is False


def test_a_successful_canary_restores_gemini_first() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    later = 1003.0 + breaker.COOLDOWN_SECONDS
    assert breaker.should_probe(breaker.GEMINI, now=later) is True
    breaker.note_success(breaker.GEMINI, now=later)

    assert breaker.state_of(breaker.GEMINI, now=later) == breaker.CLOSED
    assert breaker.preferred_order(now=later) == (breaker.GEMINI, breaker.AZURE)


def test_a_failed_canary_reopens_the_circuit_at_once() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    later = 1003.0 + breaker.COOLDOWN_SECONDS
    assert breaker.should_probe(breaker.GEMINI, now=later) is True
    breaker.note_failure(breaker.GEMINI, "still down", now=later)

    assert breaker.state_of(breaker.GEMINI, now=later) == breaker.OPEN
    assert breaker.preferred_order(now=later) == (breaker.AZURE, breaker.GEMINI)


# ---------------------------------------------------------------- 5. execution


def test_a_healthy_turn_runs_on_gemini_only() -> None:
    calls: list[str] = []

    async def gemini():
        calls.append("gemini")
        return "answer from gemini"

    async def azure():
        calls.append("azure")
        return "answer from azure"

    answer = asyncio.run(breaker.execute_turn(gemini, azure))

    assert answer == "answer from gemini"
    assert calls == ["gemini"], "Azure must not be called while Gemini is healthy"


def test_an_open_circuit_runs_on_azure_only() -> None:
    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    calls: list[str] = []

    async def gemini():
        calls.append("gemini")
        return "gemini"

    async def azure():
        calls.append("azure")
        return "azure"

    answer = asyncio.run(breaker.execute_turn(gemini, azure, now=1003.0))

    assert answer == "azure"
    assert calls == ["azure"], "an open circuit must not pay for Gemini's timeout again"


def test_a_failing_gemini_falls_back_and_records_both() -> None:
    async def gemini():
        raise RuntimeError("429 rate limited")

    async def azure():
        return "azure answer"

    answer = asyncio.run(breaker.execute_turn(gemini, azure))

    assert answer == "azure answer"
    assert breaker.state_of(breaker.AZURE) == breaker.CLOSED


def test_the_labels_the_gateway_uses_are_understood() -> None:
    assert breaker.provider_for("Gemini") == breaker.GEMINI
    assert breaker.provider_for("Azure OpenAI") == breaker.AZURE


# ---------------------------------------------------------------- 6. observability


def test_the_snapshot_names_the_state_without_naming_a_prompt() -> None:
    breaker.note_failure(breaker.GEMINI, "429 rate limited", now=1000.0)

    snapshot = breaker.snapshot(now=1000.0)

    assert snapshot[breaker.GEMINI]["state"] == breaker.CLOSED
    assert snapshot[breaker.GEMINI]["consecutive_failures"] == 1
    assert "429" in str(snapshot[breaker.GEMINI]["last_error"])
    assert set(snapshot) == {breaker.GEMINI, breaker.AZURE}


def test_the_breaker_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CIRCUIT_BREAKER_ENABLED", "false")

    for index in range(3):
        breaker.note_failure(breaker.GEMINI, "429", now=1000.0 + index)

    assert breaker.preferred_order(now=1003.0) == (breaker.GEMINI, breaker.AZURE)
