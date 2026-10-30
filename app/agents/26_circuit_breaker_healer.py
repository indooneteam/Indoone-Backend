"""Agent 26 - the healer: find out whether Gemini is back, without asking a user.

Agent 06 opens the circuit when Gemini fails and sends traffic to Azure. That is the right
call at the time and it creates the next problem: **something has to decide when to go back**,
and every wrong way of deciding costs somebody something.

* Try real traffic? The user who happens to arrive first pays for the experiment with an
  error message.
* Stay on Azure? It works, and it is the more expensive provider, so a circuit that never
  closes quietly becomes a permanent cost increase.

So the recovery is a background probe: the smallest possible request, at intervals that grow
while the provider is still down and shrink once it starts answering, and **three successes in
a row** before the circuit is allowed back. One success is a fluke; three is a provider.

The jitter is not decoration. Every instance of this service shares the same recovery clock,
and without jitter they all probe at the same instant - the provider that just came back is
immediately hammered by its own recovery.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

IDLE = "IDLE"
PROBING_ACTIVE = "PROBING_ACTIVE"
RECOVERED = "RECOVERED"

GEMINI = "gemini"


@dataclass(slots=True)
class ProbeConfig:
    required_consecutive_successes: int = 3
    initial_backoff_sec: float = 5.0
    multiplier: float = 2.0
    max_backoff_sec: float = 120.0
    jitter_pct: float = 0.10
    confirmation_interval_sec: float = 2.0


@dataclass(slots=True)
class HealerTelemetry:
    state: str = IDLE
    consecutive_successes: int = 0
    total_probes_dispatched: int = 0
    current_backoff_sec: float = 0.0
    next_probe_epoch: float = 0.0
    is_circuit_reset_ready: bool = False
    last_probe_latency_ms: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_CIRCUIT_HEALER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def calculate_backoff_with_jitter(
    base_sec: float,
    mult: float,
    max_sec: float,
    jitter: float,
    seed: Optional[int] = None,
) -> float:
    """The next delay, bounded, and never negative.

    The cap is applied to the *base* value before the jitter, so a jittered delay can exceed
    the ceiling by at most the jitter percentage - capping after jitter would make every
    instance converge on exactly the ceiling and lose the de-synchronisation the jitter is
    there for.
    """
    base = max(0.0, float(base_sec))
    multiplier = max(1.0, float(mult))
    ceiling = max(base, float(max_sec))
    spread = max(0.0, min(1.0, float(jitter)))

    grown = min(ceiling, base * multiplier)

    if spread <= 0:
        return round(grown, 3)

    generator = random.Random(seed) if seed is not None else random
    offset = generator.uniform(-spread, spread)

    return round(max(0.001, grown * (1.0 + offset)), 3)


class CircuitBreakerHealer:
    """The recovery scheduler for one provider."""

    def __init__(self, config: Optional[ProbeConfig] = None, provider: str = GEMINI) -> None:
        self.config = config or ProbeConfig()
        self.provider = provider
        self._state = IDLE
        self._consecutive_successes = 0
        self._total_probes = 0
        self._current_backoff = float(self.config.initial_backoff_sec)
        self._next_probe_epoch = 0.0
        self._reset_ready = False
        self._last_latency_ms = 0.0

    # ---------------------------------------------------------------- the circuit

    def notify_circuit_opened(self, provider: str = GEMINI, now_epoch: Optional[float] = None) -> None:
        """Agent 06 tripped. Start looking for a way back."""
        moment = time.time() if now_epoch is None else float(now_epoch)

        self.provider = provider or self.provider
        self._state = PROBING_ACTIVE
        self._consecutive_successes = 0
        self._reset_ready = False
        self._current_backoff = float(self.config.initial_backoff_sec)
        self._next_probe_epoch = moment + self._current_backoff

    def should_dispatch_probe(self, now_epoch: Optional[float] = None) -> bool:
        """Whether it is time to try. Never true when the primary is healthy."""
        moment = time.time() if now_epoch is None else float(now_epoch)

        if self._state != PROBING_ACTIVE:
            return False

        return moment >= self._next_probe_epoch

    def record_probe_result(
        self,
        success: bool,
        latency_ms: float = 0.0,
        now_epoch: Optional[float] = None,
    ) -> HealerTelemetry:
        """Record what the probe saw, and schedule the next one."""
        moment = time.time() if now_epoch is None else float(now_epoch)

        self._total_probes += 1
        self._last_latency_ms = round(max(0.0, float(latency_ms)), 2)

        if success:
            self._consecutive_successes += 1

            if self._consecutive_successes >= max(1, int(self.config.required_consecutive_successes)):
                # Three in a row. The circuit may close; agent 06 does the closing.
                self._state = RECOVERED
                self._reset_ready = True
                self._next_probe_epoch = 0.0
            else:
                # Answering but not yet trusted: confirm quickly, not at backoff speed.
                self._current_backoff = float(self.config.initial_backoff_sec)
                self._next_probe_epoch = moment + max(0.1, float(self.config.confirmation_interval_sec))
        else:
            # One failure is enough to distrust every success so far - a provider that
            # answers twice and then fails is not recovering, it is flapping.
            self._consecutive_successes = 0
            self._reset_ready = False
            self._current_backoff = calculate_backoff_with_jitter(
                self._current_backoff,
                self.config.multiplier,
                self.config.max_backoff_sec,
                self.config.jitter_pct,
            )
            self._next_probe_epoch = moment + self._current_backoff

        return self.get_telemetry(moment)

    def acknowledge_circuit_reset(self) -> None:
        """Agent 06 has closed the circuit. Back to watching."""
        self._state = IDLE
        self._consecutive_successes = 0
        self._reset_ready = False
        self._current_backoff = float(self.config.initial_backoff_sec)
        self._next_probe_epoch = 0.0

    def get_telemetry(self, now_epoch: Optional[float] = None) -> HealerTelemetry:
        return HealerTelemetry(
            state=self._state,
            consecutive_successes=self._consecutive_successes,
            total_probes_dispatched=self._total_probes,
            current_backoff_sec=round(self._current_backoff, 3),
            next_probe_epoch=round(self._next_probe_epoch, 3),
            is_circuit_reset_ready=self._reset_ready,
            last_probe_latency_ms=self._last_latency_ms,
        )

    # ---------------------------------------------------------------- agent 06

    def sync_from_circuit(self, circuit: Any, now_epoch: Optional[float] = None) -> str:
        """Follow agent 06's state rather than duplicating it.

        The breaker is the authority on whether traffic may go to the primary. This only
        decides *when to ask*; if the breaker closed the circuit on its own canary, the healer
        stops probing instead of arguing with it.
        """
        try:
            state = str(circuit.state_of(self.provider) or "").upper()
        except Exception:
            return self._state

        if state == "CLOSED" and self._state != IDLE:
            self.acknowledge_circuit_reset()
        elif state in {"OPEN", "HALF_OPEN"} and self._state == IDLE:
            self.notify_circuit_opened(self.provider, now_epoch)

        return self._state


def action_chips(telemetry: HealerTelemetry) -> list[dict[str, str]]:
    if telemetry.state == RECOVERED:
        return [{"label": "View Fallback Recovery Log", "action": "view_recovery_log"}]

    if telemetry.state == PROBING_ACTIVE:
        return [{"label": "Check Provider Health", "action": "check_provider_health"}]

    return []
