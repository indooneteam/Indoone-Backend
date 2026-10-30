"""Agent 06 - the Circuit Breaker: Gemini first, Azure when Gemini is unwell.

The provider order used to be decided in two places, differently: the blocking path asked
"which backend is selected" and put that one first, and the streaming path asked "is Azure
enabled" and put Azure first. Both are the same mistake - a preference expressed as an `if`
in the caller - and both mean a deployment that selects Azure never tries Gemini at all.

This agent owns that decision instead:

    CLOSED     Gemini first. Every turn goes to Gemini; Azure waits.
    OPEN       Gemini tripped. Every turn goes straight to Azure, with no wait: the cost of
               a dead primary is paid once, not on every request.
    HALF_OPEN  The cooldown passed. One canary goes to Gemini; success closes the circuit,
               failure re-opens it for another cooldown.

The trip is three consecutive failures inside a sixty-second window - 429, 5xx, timeout or a
socket error, all the same thing from here: the primary did not answer. Three rather than
one because a single blip is not an outage; consecutive rather than a count, because a
provider that fails once an hour is healthy.

Cooldown is thirty seconds, which is long enough for a rate-limit window to move and short
enough that a recovered provider is back before a person notices.

State is per process, in a bounded structure, and every decision is arithmetic. Nothing here
imports a provider, opens a socket or reads a file, so the machine can be tested without
either service being up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

GEMINI = "gemini"
AZURE = "azure"

CLOSED = "CLOSED"
OPEN = "OPEN"
HALF_OPEN = "HALF_OPEN"

# Three consecutive failures inside a minute trips the circuit.
FAILURE_THRESHOLD = 3
FAILURE_WINDOW_SECONDS = 60.0
COOLDOWN_SECONDS = 30.0

# Labels the gateway uses, mapped to the keys here, so neither side has to guess.
LABELS = {"Gemini": GEMINI, "Azure OpenAI": AZURE}


@dataclass(slots=True)
class CircuitState:
    """What is remembered about one provider."""

    failures: int = 0
    first_failure_at: float = 0.0
    opened_at: float = 0.0
    last_error: str = ""
    canary_pending: bool = False
    trips: int = 0


_state: dict[str, CircuitState] = {GEMINI: CircuitState(), AZURE: CircuitState()}


def reset() -> None:
    """Forget every failure. Used by tests, and by an operator after a fix."""
    for key in (GEMINI, AZURE):
        _state[key] = CircuitState()


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_CIRCUIT_BREAKER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _now(now: float | None) -> float:
    return time.monotonic() if now is None else float(now)


def state_of(provider: str, *, now: float | None = None) -> str:
    """The state of one provider, accounting for a cooldown that has since elapsed."""
    circuit = _state.get(provider)

    if circuit is None:
        return CLOSED

    if circuit.opened_at and _now(now) - circuit.opened_at >= COOLDOWN_SECONDS:
        return HALF_OPEN

    return OPEN if circuit.opened_at else CLOSED


def note_success(provider: str, *, now: float | None = None) -> None:
    """A provider answered. Anything it had against it is cleared."""
    circuit = _state.get(provider)

    if circuit is None:
        return

    circuit.failures = 0
    circuit.first_failure_at = 0.0
    circuit.opened_at = 0.0
    circuit.canary_pending = False
    circuit.last_error = ""


def note_failure(provider: str, reason: str = "", *, now: float | None = None) -> None:
    """A provider did not answer. Three in a row inside the window opens the circuit."""
    circuit = _state.get(provider)

    if circuit is None:
        return

    moment = _now(now)
    circuit.last_error = str(reason)[:200]

    # A failure outside the window starts the count again: a provider that fails once an
    # hour is not degrading, it is unlucky.
    if circuit.first_failure_at and moment - circuit.first_failure_at > FAILURE_WINDOW_SECONDS:
        circuit.failures = 0
        circuit.first_failure_at = 0.0

    if not circuit.first_failure_at:
        circuit.first_failure_at = moment

    circuit.failures += 1

    # A failed canary re-opens immediately: the cooldown already passed once.
    if circuit.canary_pending:
        circuit.canary_pending = False
        circuit.opened_at = moment
        circuit.trips += 1
        return

    if circuit.failures >= FAILURE_THRESHOLD:
        circuit.opened_at = moment
        circuit.trips += 1


def preferred_order(*, now: float | None = None) -> tuple[str, str]:
    """Which provider to try first, and which to fall back to.

    Gemini first is the policy, not a coincidence of configuration. The only thing that
    moves Azure ahead of it is Gemini being OPEN.
    """
    if not enabled():
        return (GEMINI, AZURE)

    if state_of(GEMINI, now=now) == OPEN:
        return (AZURE, GEMINI)

    return (GEMINI, AZURE)


def should_probe(provider: str, *, now: float | None = None) -> bool:
    """Whether this turn is the canary for a provider in HALF_OPEN."""
    if state_of(provider, now=now) != HALF_OPEN:
        return False

    circuit = _state[provider]

    if circuit.canary_pending:
        return False

    circuit.canary_pending = True
    return True


def provider_for(label: str) -> str:
    """Translate the gateway's display label into the key used here."""
    return LABELS.get(label, label.strip().casefold())


def snapshot(*, now: float | None = None) -> dict[str, dict[str, object]]:
    """What an operator needs to see, and nothing about a prompt."""
    return {
        provider: {
            "state": state_of(provider, now=now),
            "consecutive_failures": circuit.failures,
            "trips": circuit.trips,
            "last_error": circuit.last_error,
        }
        for provider, circuit in _state.items()
    }


async def execute_turn(primary, secondary, *, now: float | None = None):
    """Run a blocking turn through the breaker.

    ``primary`` and ``secondary`` are callables returning the answer. The order comes from
    ``preferred_order``, so the caller never decides - which is the whole point: two callers
    deciding is how the streaming path and the blocking path came to disagree.
    """
    first, second = preferred_order(now=now)
    calls = {GEMINI: primary, AZURE: secondary}

    try:
        answer = await calls[first]()
    except Exception as exc:
        note_failure(first, str(exc), now=now)
        answer = await calls[second]()
        note_success(second, now=now)
        return answer

    note_success(first, now=now)
    return answer
