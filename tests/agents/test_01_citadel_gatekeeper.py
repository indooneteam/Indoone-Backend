"""Agent 01 - the Citadel Gatekeeper.

The contract these tests hold, in the order the agent checks:

1. A human, typing at human speed, passes - and costs under a millisecond.
2. A script firing ten requests in a second does not.
3. A flood of one character is refused before it reaches a model.
4. An oversized payload is refused before anything downstream is built.
5. Nothing it remembers grows without a ceiling, however many clients arrive.
"""

from __future__ import annotations

import time

import pytest

from app.agents import gatekeeper

agent = gatekeeper()

PASS = agent.PASS
THROTTLE = agent.THROTTLE
BLOCK = agent.BLOCK


@pytest.fixture(autouse=True)
def _clean_state():
    agent.reset()
    yield
    agent.reset()


def _human_turn(subject: str, moment: float, text: str = "when is deepavali this year?") -> str:
    return agent.evaluate(subject=subject, payload=text, now=moment).status


# ---------------------------------------------------------------- 1. humans pass


def test_a_single_human_message_passes() -> None:
    decision = agent.evaluate(subject="user-1", payload="when is deepavali?", now=1000.0)

    assert decision.status == PASS
    assert decision.threat_vector == agent.CLEAN
    assert decision.confidence_score == 0.0
    assert decision.retry_after == 0
    assert decision.allowed is True


def test_a_human_conversation_passes_turn_after_turn() -> None:
    """A person typing every few seconds must never be throttled by this agent."""
    moment = 1000.0
    for index in range(20):
        moment += 4.0
        decision = agent.evaluate(
            subject="user-2", payload=f"turn {index}: and what about the next one?", now=moment
        )
        assert decision.status == PASS, f"turn {index} was refused"


def test_a_normal_turn_costs_under_a_millisecond() -> None:
    """Measured as a mean over many calls: a single sample on a shared CI box is noise."""
    agent.reset()
    payload = "web research maadu jio ipo estakke listing agide"
    moment = 5000.0

    started = time.perf_counter()
    for index in range(200):
        moment += 3.0
        agent.evaluate(subject=f"perf-{index}", payload=payload, now=moment)
    elapsed_ms = (time.perf_counter() - started) * 1000 / 200

    assert elapsed_ms < 1.0, f"mean latency {elapsed_ms:.3f} ms"


# ---------------------------------------------------------------- 2. bursts do not


def test_a_rapid_burst_is_throttled_or_blocked() -> None:
    moment = 2000.0
    verdicts = []

    for _ in range(40):
        moment += 0.01  # ten a second
        verdicts.append(agent.evaluate(subject="bot-1", payload="tell me something", now=moment).status)

    assert verdicts[0] == PASS, "the first request must be allowed"
    assert THROTTLE in verdicts or BLOCK in verdicts
    assert verdicts.count(PASS) < len(verdicts)


def test_the_bucket_widens_for_a_steady_cadence() -> None:
    """The adaptive half: same request count, different cadence, different verdicts."""
    fast_moment, slow_moment = 3000.0, 3000.0
    fast, slow = 0, 0

    for _ in range(30):
        fast_moment += 0.2
        if agent.evaluate(subject="fast", payload="question", now=fast_moment).status == PASS:
            fast += 1

    for _ in range(30):
        slow_moment += 3.0
        if agent.evaluate(subject="slow", payload="question", now=slow_moment).status == PASS:
            slow += 1

    assert slow > fast


def test_repeated_violations_escalate_to_a_block_with_backoff() -> None:
    moment = 4000.0
    seen_block = False

    for _ in range(60):
        moment += 0.01
        decision = agent.evaluate(subject="repeat-offender", payload="spam me", now=moment)
        if decision.status == BLOCK:
            seen_block = True
            assert decision.retry_after > 0, "a block must say when it lifts"
            break

    assert seen_block


def test_a_block_lifts_by_itself() -> None:
    moment = 6000.0
    for _ in range(60):
        moment += 0.01
        decision = agent.evaluate(subject="serving-time", payload="spam", now=moment)
        if decision.status == BLOCK:
            break

    later = moment + decision.retry_after + 1
    assert agent.evaluate(subject="serving-time", payload="a normal question", now=later).status == PASS


# ---------------------------------------------------------------- 3. shape


@pytest.mark.parametrize(
    "payload",
    [
        "a" * 500,
        "!" * 300,
        " " * 400,
        "ab" * 200,
    ],
)
def test_a_zero_entropy_flood_is_refused(payload: str) -> None:
    decision = agent.evaluate(subject="flood", payload=payload, now=7000.0)

    assert decision.status in {THROTTLE, BLOCK}
    assert decision.threat_vector == agent.ENTROPY_ANOMALY


def test_ordinary_text_is_not_mistaken_for_a_flood() -> None:
    text = (
        "Namaskara, naanu Jio IPO bagge keluttidini. Listing date announce aagideya? "
        "Official SEBI filing ideya antha nodi heli."
    )

    assert agent.evaluate(subject="reader", payload=text, now=8000.0).status == PASS


# ---------------------------------------------------------------- 4. size


def test_an_oversized_payload_is_refused() -> None:
    decision = agent.evaluate(subject="big", payload="x" * (agent.MAX_PAYLOAD_BYTES + 1), now=9000.0)

    assert decision.status == BLOCK
    assert decision.threat_vector == agent.OVERSIZED_PAYLOAD
    assert decision.confidence_score == 1.0


def test_an_oversized_payload_is_refused_before_it_is_remembered() -> None:
    """Refused early means no fingerprint, no bucket spend, no growth."""
    before = agent.stats()
    agent.evaluate(subject="big-2", payload="y" * (agent.MAX_PAYLOAD_BYTES + 1), now=9100.0)
    after = agent.stats()

    assert after["fingerprints"] == before["fingerprints"]


# ---------------------------------------------------------------- 5. memory


def test_state_stays_bounded_under_ten_thousand_clients() -> None:
    for index in range(10_000):
        agent.evaluate(subject=f"client-{index}", payload=f"question {index}", now=10_000.0 + index * 0.001)

    sizes = agent.stats()

    assert sizes["clients"] <= agent.MAX_CLIENTS
    assert sizes["fingerprints"] <= agent.MAX_FINGERPRINTS


def test_one_client_cannot_grow_its_own_record_without_a_ceiling() -> None:
    for index in range(500):
        agent.evaluate(subject="chatty", payload=f"distinct message number {index}", now=20_000.0 + index)

    state = agent._clients["chatty"]

    assert len(state.repeats) <= agent.REPEAT_LIMIT * 8


def test_the_decision_is_bilingual() -> None:
    decision = agent.evaluate(subject="kn", payload="a" * 400, now=30_000.0)

    assert decision.client_message
    assert decision.client_message_kn
    assert any("\u0c80" <= character <= "\u0cff" for character in decision.client_message_kn)


def test_the_telemetry_is_always_present() -> None:
    decision = agent.evaluate(subject="tel", payload="hello", now=40_000.0)

    assert decision.telemetry.elapsed_microseconds >= 0
    assert 0.0 <= decision.telemetry.bucket_pressure <= 1.0
    assert 0.0 <= decision.telemetry.reputation <= 1.0


# ---------------------------------------------------------------- 6. wiring


def _auth_headers(user_id: str = "gate-user", secret: str = "x" * 32) -> dict[str, str]:
    """The same signed bearer the app issues, so this drives the real endpoint."""
    import base64
    import hashlib
    import hmac
    import time as _time

    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(user_id.encode("utf-8"))
    stamp = encode(str(int(_time.time())).encode("ascii"))
    signature = encode(hmac.new(secret.encode("ascii"), f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{stamp}.{signature}"}


def test_the_chat_entry_consults_the_agent(monkeypatch) -> None:
    """The gate is only real if the endpoint asks it. This drives the endpoint itself."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_GATEKEEPER_ENABLED", "true")

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return "ok"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)
    agent.reset()

    client = TestClient(app)
    headers = _auth_headers()
    statuses = []

    for _ in range(30):
        response = client.post(
            "/api/chat", json={"message": "when is deepavali this year?"}, headers=headers
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            assert "Retry-After" in response.headers
            assert response.json()["message"]
            break

    assert statuses[0] == 200, f"the first message must be allowed, got {statuses[0]}"
    assert 429 in statuses, f"a burst at the chat entry must be refused, got {statuses}"
