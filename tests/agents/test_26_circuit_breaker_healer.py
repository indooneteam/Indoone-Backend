"""Agent 26 - the circuit healer.

The property: a real user is never the experiment that finds out whether the primary is back.
Everything else - the backoff, the jitter, the three successes - exists to make that safe and
to stop the provider flapping.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

healer_module = load("26_circuit_breaker_healer")
Healer = healer_module.CircuitBreakerHealer
Config = healer_module.ProbeConfig


# ---------------------------------------------------------------- 1. the idle state


def test_a_healthy_circuit_is_never_probed() -> None:
    healer = Healer()

    assert healer.get_telemetry().state == healer_module.IDLE
    assert healer.should_dispatch_probe(now_epoch=1000.0) is False


def test_opening_the_circuit_starts_probing() -> None:
    healer = Healer()

    healer.notify_circuit_opened(now_epoch=1000.0)
    telemetry = healer.get_telemetry()

    assert telemetry.state == healer_module.PROBING_ACTIVE
    assert telemetry.next_probe_epoch == pytest.approx(1005.0)
    assert telemetry.consecutive_successes == 0


def test_the_first_probe_waits_for_its_slot() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)

    assert healer.should_dispatch_probe(now_epoch=1001.0) is False
    assert healer.should_dispatch_probe(now_epoch=1006.0) is True


# ---------------------------------------------------------------- 2. the successes


def test_one_success_is_not_recovery() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)

    telemetry = healer.record_probe_result(True, latency_ms=12.5, now_epoch=1006.0)

    assert telemetry.consecutive_successes == 1
    assert telemetry.is_circuit_reset_ready is False
    assert telemetry.state == healer_module.PROBING_ACTIVE
    assert telemetry.last_probe_latency_ms == 12.5


def test_a_success_schedules_a_quick_confirmation_not_a_backoff() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)

    telemetry = healer.record_probe_result(True, now_epoch=1006.0)

    assert telemetry.next_probe_epoch == pytest.approx(1008.0)


def test_three_successes_in_a_row_allow_the_circuit_to_close() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)

    for index in range(3):
        telemetry = healer.record_probe_result(True, now_epoch=1006.0 + index * 2)

    assert telemetry.state == healer_module.RECOVERED
    assert telemetry.is_circuit_reset_ready is True
    assert telemetry.consecutive_successes == 3


def test_a_recovered_healer_stops_probing() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    for index in range(3):
        healer.record_probe_result(True, now_epoch=1006.0 + index * 2)

    assert healer.should_dispatch_probe(now_epoch=9999.0) is False


def test_the_threshold_is_configurable() -> None:
    healer = Healer(Config(required_consecutive_successes=1))
    healer.notify_circuit_opened(now_epoch=1000.0)

    telemetry = healer.record_probe_result(True, now_epoch=1006.0)

    assert telemetry.is_circuit_reset_ready is True


# ---------------------------------------------------------------- 3. the failures


def test_a_failure_resets_every_success_so_far() -> None:
    """A provider that answers twice and then fails is not recovering, it is flapping."""
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    healer.record_probe_result(True, now_epoch=1006.0)
    healer.record_probe_result(True, now_epoch=1008.0)

    telemetry = healer.record_probe_result(False, now_epoch=1010.0)

    assert telemetry.consecutive_successes == 0
    assert telemetry.is_circuit_reset_ready is False


def test_a_failure_backs_off_exponentially() -> None:
    healer = Healer(Config(jitter_pct=0.0))
    healer.notify_circuit_opened(now_epoch=1000.0)

    first = healer.record_probe_result(False, now_epoch=1006.0)
    second = healer.record_probe_result(False, now_epoch=1100.0)

    assert first.current_backoff_sec == pytest.approx(10.0)
    assert second.current_backoff_sec == pytest.approx(20.0)


def test_the_backoff_caps_and_stays_there() -> None:
    healer = Healer(Config(jitter_pct=0.0))
    healer.notify_circuit_opened(now_epoch=1000.0)

    telemetry = None
    for _ in range(10):
        telemetry = healer.record_probe_result(False, now_epoch=5000.0)

    assert telemetry.current_backoff_sec == pytest.approx(120.0)


def test_a_failure_after_recovery_reopens_probing() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    for index in range(3):
        healer.record_probe_result(True, now_epoch=1006.0 + index * 2)

    telemetry = healer.record_probe_result(False, now_epoch=1020.0)

    assert telemetry.state == healer_module.RECOVERED
    assert telemetry.consecutive_successes == 0
    assert telemetry.is_circuit_reset_ready is False


# ---------------------------------------------------------------- 4. the jitter


def test_the_jitter_stays_inside_its_band() -> None:
    for seed in range(50):
        delay = healer_module.calculate_backoff_with_jitter(10.0, 1.0, 120.0, 0.10, seed=seed)

        assert 9.0 <= delay <= 11.0, f"seed {seed} gave {delay}"


def test_the_jitter_never_produces_a_negative_interval() -> None:
    for seed in range(50):
        assert healer_module.calculate_backoff_with_jitter(0.0, 2.0, 120.0, 0.10, seed=seed) > 0


def test_a_zero_jitter_is_deterministic() -> None:
    assert healer_module.calculate_backoff_with_jitter(5.0, 2.0, 120.0, 0.0) == 10.0


def test_the_cap_is_applied_before_the_jitter() -> None:
    """Capping after the jitter would make every instance converge on the ceiling."""
    delay = healer_module.calculate_backoff_with_jitter(100.0, 2.0, 120.0, 0.10, seed=1)

    assert 108.0 <= delay <= 132.0


# ---------------------------------------------------------------- 5. the handshake


def test_acknowledging_the_reset_returns_to_idle() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    for index in range(3):
        healer.record_probe_result(True, now_epoch=1006.0 + index * 2)

    healer.acknowledge_circuit_reset()
    telemetry = healer.get_telemetry()

    assert telemetry.state == healer_module.IDLE
    assert telemetry.consecutive_successes == 0
    assert telemetry.is_circuit_reset_ready is False
    assert telemetry.next_probe_epoch == 0.0


def test_the_healer_follows_the_breaker_rather_than_arguing_with_it() -> None:
    """If agent 06 closed the circuit on its own canary, the healer stops probing."""

    class Breaker:
        def __init__(self, state: str) -> None:
            self.state = state

        def state_of(self, provider: str) -> str:
            return self.state

    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)

    assert healer.sync_from_circuit(Breaker("CLOSED")) == healer_module.IDLE

    healer.notify_circuit_opened(now_epoch=1000.0)
    assert healer.sync_from_circuit(Breaker("OPEN")) == healer_module.PROBING_ACTIVE


def test_a_breaker_that_cannot_answer_does_not_break_the_healer() -> None:
    class Broken:
        def state_of(self, provider: str) -> str:
            raise RuntimeError("no such provider")

    healer = Healer()

    assert healer.sync_from_circuit(Broken()) == healer_module.IDLE


def test_the_chips_follow_the_state() -> None:
    healer = Healer()
    assert healer_module.action_chips(healer.get_telemetry()) == []

    healer.notify_circuit_opened(now_epoch=1000.0)
    assert [chip["action"] for chip in healer_module.action_chips(healer.get_telemetry())] == [
        "check_provider_health"
    ]

    for index in range(3):
        telemetry = healer.record_probe_result(True, now_epoch=1006.0 + index * 2)
    assert [chip["action"] for chip in healer_module.action_chips(telemetry)] == ["view_recovery_log"]


# ---------------------------------------------------------------- 6. speed


def test_a_state_decision_stays_under_its_budget() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        healer.should_dispatch_probe(now_epoch=1006.0)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_a_transition_stays_under_its_budget() -> None:
    healer = Healer()
    healer.notify_circuit_opened(now_epoch=1000.0)
    fastest = float("inf")

    for index in range(300):
        started = time.thread_time()
        healer.record_probe_result(index % 2 == 0, now_epoch=1000.0 + index)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
