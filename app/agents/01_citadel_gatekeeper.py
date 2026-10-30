"""Agent 01 - the Citadel Gatekeeper: an ingress defender that thinks before it refuses.

A rate limiter counts. This decides. It reads the same request three ways - how much, how
often, and what it looks like - and returns a typed verdict rather than a bare 429:

* **tempo** - an adaptive leaky bucket whose burst allowance follows the client's own
  cadence, so a human who types every few seconds gets a wider bucket than a script that
  fires ten requests in a second.
* **shape** - zero-entropy floods (one character repeated, whitespace storms) and
  oversized payloads are refused before any buffer is built downstream.
* **repetition** - identical payloads hashed into a bounded ring, so a bot replaying one
  query is caught while a user retrying the same message twice is not.
* **penalty** - repeated violations escalate on a Fibonacci backoff, from a soft throttle
  to a temporary block that expires on its own.

Everything it keeps is bounded: 4096 client records and 4096 fingerprints, LRU-evicted.
There is no thread, no timer, no I/O, and no dependency outside the standard library, so
a call costs arithmetic and one hash.

It answers with ``Decision``, which carries the verdict, a confidence, the threat vector,
a message the client can show a person in English or Kannada, and telemetry.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

# ---------------------------------------------------------------- bounds

MAX_CLIENTS = 4096
MAX_FINGERPRINTS = 4096
MAX_PAYLOAD_BYTES = 32_768
FINGERPRINT_TTL_SECONDS = 30.0
REPEAT_LIMIT = 3  # identical payloads inside the window before it is a flood

BASE_BURST = 8  # what an unknown client starts with
BUCKET_REFILL_PER_SECOND = 0.5  # one token every two seconds, per client
MAX_BURST_BONUS = 6  # the widest a well-behaved cadence can widen the bucket

STRIKE_THROTTLE = 1  # first violation: throttle
STRIKES_BEFORE_BLOCK = 3  # third violation inside the window: block
STRIKE_WINDOW_SECONDS = 300.0
FIBONACCI_BASE_SECONDS = 5  # 5, 5, 10, 15, 25 ... capped
FIBONACCI_CAP_SECONDS = 300

# A single character occupying this share of a long payload is a flood, not a sentence.
_FLOOD_MIN_LENGTH = 48
_FLOOD_SHARE = 0.6
_LOW_VARIETY = 3

PASS = "PASS"
THROTTLE = "THROTTLE"
BLOCK = "BLOCK"

CLEAN = "CLEAN"
TEMPO_SPIKE = "TEMPO_SPIKE"
SPAM_FLOOD = "SPAM_FLOOD"
ENTROPY_ANOMALY = "ENTROPY_ANOMALY"
OVERSIZED_PAYLOAD = "OVERSIZED_PAYLOAD"

_MESSAGES = {
    TEMPO_SPIKE: (
        "ಸ್ವಲ್ಪ ನಿಧಾನವಾಗಿ - ಹಲವು ಸಂದೇಶಗಳು ಒಟ್ಟಿಗೆ ಬಂದಿವೆ. {retry} ಸೆಕೆಂಡ್ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "A little slower please - several messages arrived at once. Try again in {retry}s.",
    ),
    SPAM_FLOOD: (
        "ಅದೇ ಸಂದೇಶ ಪುನಃ ಪುನಃ ಬರುತ್ತಿದೆ. ಒಂದು ಬಾರಿ ಕಳಿಸಿ, ಉತ್ತರಕ್ಕೆ ಕಾಯಿರಿ.",
        "The same message keeps arriving. Send it once and wait for the reply.",
    ),
    ENTROPY_ANOMALY: (
        "ಸಂದೇಶವನ್ನು ಓದಲಾಗುತ್ತಿಲ್ಲ - ಅರ್ಥವಿರುವ ಪಠ್ಯ ಕಳಿಸಿ.",
        "That message could not be read - send it as ordinary text.",
    ),
    OVERSIZED_PAYLOAD: (
        "ಸಂದೇಶ ತುಂಬಾ ದೊಡ್ಡದಾಗಿದೆ. ಸ್ವಲ್ಪ ಚಿಕ್ಕದಾಗಿ ಕಳಿಸಿ.",
        "That message is too large. Send a shorter one.",
    ),
    CLEAN: ("", ""),
}


# ---------------------------------------------------------------- contracts


@dataclass(slots=True, frozen=True)
class Telemetry:
    """What the operator needs to see after a decision, and nothing else."""

    elapsed_microseconds: int
    bucket_pressure: float  # 0.0 empty, 1.0 full
    reputation: float  # 0.0 hostile, 1.0 untouched
    strikes: int
    tracked_clients: int
    tracked_fingerprints: int


@dataclass(slots=True, frozen=True)
class Decision:
    """The verdict. One of PASS, THROTTLE, BLOCK - never a bare boolean."""

    status: str
    confidence_score: float
    threat_vector: str
    client_message: str
    client_message_kn: str
    retry_after: int
    telemetry: Telemetry

    @property
    def allowed(self) -> bool:
        return self.status == PASS

    @property
    def blocked(self) -> bool:
        return self.status == BLOCK


@dataclass(slots=True)
class _ClientState:
    tokens: float
    updated: float
    last_seen: float
    strikes: int = 0
    window_start: float = 0.0
    mean_interval: float = 0.0
    samples: int = 0
    blocked_until: float = 0.0
    repeats: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------- state

_clients: OrderedDict[str, _ClientState] = OrderedDict()
_fingerprints: OrderedDict[int, float] = OrderedDict()


def reset() -> None:
    """Drop every record. Used by tests and by a deliberate operator reset."""
    _clients.clear()
    _fingerprints.clear()


def stats() -> dict[str, int]:
    """Sizes of the bounded structures, for tests and for /health style reporting."""
    return {"clients": len(_clients), "fingerprints": len(_fingerprints)}


# ---------------------------------------------------------------- helpers


def _evict(store: OrderedDict, cap: int) -> None:
    while len(store) > cap:
        store.popitem(last=False)


def _fingerprint(payload: str) -> int:
    return int.from_bytes(hashlib.blake2b(payload.encode("utf-8", "ignore"), digest_size=8).digest(), "big")


def _flood(payload: str) -> bool:
    """Zero-entropy floods: one character repeated, or almost no variety at all."""
    if len(payload) < _FLOOD_MIN_LENGTH:
        return False

    counts: dict[str, int] = {}
    for character in payload:
        counts[character] = counts.get(character, 0) + 1

    if not counts:
        return False

    if max(counts.values()) / len(payload) >= _FLOOD_SHARE:
        return True

    # "abababab..." - few distinct characters over a long payload.
    return len([c for c in counts if not c.isspace()]) <= _LOW_VARIETY and len(payload) >= _FLOOD_MIN_LENGTH * 2


def _fibonacci(step: int) -> int:
    a, b = 1, 1
    for _ in range(max(0, step)):
        a, b = b, a + b
    return min(FIBONACCI_CAP_SECONDS, FIBONACCI_BASE_SECONDS * a)


def _allowance(state: _ClientState) -> float:
    """The bucket size this client has earned.

    A steady cadence widens it; a silent client falls back to the base. This is what makes
    it adaptive rather than a fixed integer: a person typing every four seconds should not
    be throttled by a budget designed for a script.
    """
    if state.samples < 2 or state.mean_interval <= 0:
        return float(BASE_BURST)

    bonus = min(MAX_BURST_BONUS, int(state.mean_interval / 2))
    return float(BASE_BURST + bonus)


def _retry_after(state: _ClientState, moment: float, status: str, bucket_retry: float) -> float:
    """How long the client must wait, in the units of the verdict.

    A BLOCK waits for the block to lift - that is the sentence it was given. Anything else
    waits only for a token to come back. Mixing the two told a blocked client to retry in
    one second while its block still had five to run, which is a worse experience than a
    plain refusal: it invites a request that is certain to fail.
    """
    if status == BLOCK and state.blocked_until > moment:
        return state.blocked_until - moment
    return bucket_retry


def _message(vector: str, retry: float) -> tuple[str, str]:
    """A message a person reads: whole seconds, rounded up, never a float."""
    seconds = max(1, int(-(-retry // 1))) if retry else 0
    kn, en = _MESSAGES.get(vector, _MESSAGES[CLEAN])
    return kn.format(retry=seconds), en.format(retry=seconds)


def _decision(
    *,
    status: str,
    vector: str,
    confidence: float,
    retry: int,
    state: _ClientState,
    started: float,
    pressure: float,
) -> Decision:
    kn, en = _message(vector, retry)
    return Decision(
        status=status,
        confidence_score=round(min(1.0, max(0.0, confidence)), 3),
        threat_vector=vector,
        client_message=en,
        client_message_kn=kn,
        retry_after=max(0, int(retry)),
        telemetry=Telemetry(
            elapsed_microseconds=int((time.perf_counter() - started) * 1_000_000),
            bucket_pressure=round(min(1.0, max(0.0, pressure)), 3),
            reputation=round(max(0.0, 1.0 - (state.strikes / 10.0)), 3),
            strikes=state.strikes,
            tracked_clients=len(_clients),
            tracked_fingerprints=len(_fingerprints),
        ),
    )


def _strike(state: _ClientState, now: float) -> None:
    """Record a violation, and escalate if they are piling up inside the window."""
    if state.window_start == 0.0 or now - state.window_start > STRIKE_WINDOW_SECONDS:
        state.window_start = now
        state.strikes = 0

    state.strikes += 1

    if state.strikes >= STRIKES_BEFORE_BLOCK:
        state.blocked_until = now + _fibonacci(state.strikes - STRIKES_BEFORE_BLOCK)


# ---------------------------------------------------------------- the gate


def evaluate(
    *,
    subject: str,
    payload: str,
    client_ip: str = "",
    now: float | None = None,
) -> Decision:
    """Decide one ingress request.

    ``subject`` is the authenticated user id when there is one; otherwise the caller passes
    the client address, so an unauthenticated flood is still bounded by something.
    """
    started = time.perf_counter()
    moment = time.monotonic() if now is None else float(now)
    text = payload or ""

    key = (subject or client_ip or "anonymous").strip() or "anonymous"

    state = _clients.get(key)
    if state is None:
        state = _ClientState(tokens=float(BASE_BURST), updated=moment, last_seen=moment)
        _clients[key] = state
    _clients.move_to_end(key)
    _evict(_clients, MAX_CLIENTS)

    allowance = _allowance(state)

    # 1. An expired block is over, and it ends clean: the sentence was served, so the
    # strikes go with it. Without this a block would renew itself on the next violation
    # and a client could never come back - a permanent ban decided by arithmetic, which
    # is exactly the thing this agent must not do on its own.
    if state.blocked_until and state.blocked_until <= moment:
        state.blocked_until = 0.0
        state.strikes = 0
        state.window_start = 0.0
        state.tokens = allowance
        state.updated = moment

    # Anything still inside its sentence waits.
    if state.blocked_until > moment:
        return _decision(
            status=BLOCK,
            vector=TEMPO_SPIKE,
            confidence=0.95,
            retry=state.blocked_until - moment,
            state=state,
            started=started,
            pressure=1.0,
        )

    # 2. Size before shape: an oversized payload is refused before anything is scanned.
    if len(text.encode("utf-8", "ignore")) > MAX_PAYLOAD_BYTES:
        _strike(state, moment)
        return _decision(
            status=BLOCK,
            vector=OVERSIZED_PAYLOAD,
            confidence=1.0,
            retry=0,
            state=state,
            started=started,
            pressure=1.0,
        )

    # 3. Shape: a flood of one character is not a question.
    if _flood(text):
        _strike(state, moment)
        status = BLOCK if state.strikes >= STRIKES_BEFORE_BLOCK else THROTTLE
        return _decision(
            status=status,
            vector=ENTROPY_ANOMALY,
            confidence=0.9,
            retry=_retry_after(state, moment, status, _fibonacci(max(0, state.strikes - 1))),
            state=state,
            started=started,
            pressure=1.0,
        )

    # 4. Repetition: the same payload inside the window, from a client already at speed.
    if text.strip():
        digest = _fingerprint(text.strip())
        seen_at = state.repeats.get(digest)
        if seen_at is not None and moment - seen_at < FINGERPRINT_TTL_SECONDS:
            count = sum(1 for value in state.repeats.values() if moment - value < FINGERPRINT_TTL_SECONDS)
            if count >= REPEAT_LIMIT:
                _strike(state, moment)
                return _decision(
                    status=THROTTLE,
                    vector=SPAM_FLOOD,
                    confidence=0.8,
                    retry=_retry_after(state, moment, THROTTLE, _fibonacci(max(0, state.strikes - 1))),
                    state=state,
                    started=started,
                    pressure=1.0,
                )
        state.repeats[digest] = moment
        # bounded per client as well as globally
        while len(state.repeats) > REPEAT_LIMIT * 8:
            state.repeats.pop(next(iter(state.repeats)))
        _fingerprints[digest] = moment
        _fingerprints.move_to_end(digest)
        _evict(_fingerprints, MAX_FINGERPRINTS)

    # 5. Tempo: refill, then spend. A human cadence has already widened the bucket.
    elapsed = max(0.0, moment - state.updated)
    state.tokens = min(allowance, state.tokens + elapsed * BUCKET_REFILL_PER_SECOND)
    state.updated = moment

    interval = moment - state.last_seen
    if interval > 0:
        state.samples += 1
        state.mean_interval = interval if state.samples == 1 else (state.mean_interval * 0.7 + interval * 0.3)
    state.last_seen = moment

    if state.tokens < 1.0:
        _strike(state, moment)
        bucket_retry = max(1.0, (1.0 - state.tokens) / BUCKET_REFILL_PER_SECOND)
        status = BLOCK if state.strikes >= STRIKES_BEFORE_BLOCK else THROTTLE
        return _decision(
            status=status,
            vector=TEMPO_SPIKE,
            confidence=0.7 + min(0.2, state.strikes * 0.05),
            retry=_retry_after(state, moment, status, bucket_retry),
            state=state,
            started=started,
            pressure=1.0,
        )

    state.tokens -= 1.0

    return _decision(
        status=PASS,
        vector=CLEAN,
        confidence=0.0,
        retry=0,
        state=state,
        started=started,
        pressure=1.0 - (state.tokens / allowance if allowance else 1.0),
    )


# ---------------------------------------------------------------- wiring


def enabled() -> bool:
    """Whether the app should consult this agent.

    On by default in a running service; the test suite sets INDOONE_GATEKEEPER_ENABLED to
    false so a rate decision never depends on how many requests a test happened to make.
    """
    import os

    return os.getenv("INDOONE_GATEKEEPER_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
