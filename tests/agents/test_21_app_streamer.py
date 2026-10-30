"""Agent 21 - the app streamer.

Two properties that cost real money when they fail: the wire format must be exactly what a
client parses, and a client that leaves must stop the model rather than let it talk to an
empty socket.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.agents import load

streamer = load("21_app_streamer")
Supervisor = streamer.AppStreamSupervisor
StreamEvent = streamer.StreamEvent


def frames(feed, **kwargs) -> list[str]:
    async def collect():
        return [frame async for frame in Supervisor.bridge_stream(feed, **kwargs)]

    return asyncio.run(collect())


async def tokens(*chunks, delay: float = 0.0):
    for chunk in chunks:
        if delay:
            await asyncio.sleep(delay)
        yield chunk


def parse(frame: str) -> tuple[str, dict]:
    """(event, data) from one SSE frame."""
    lines = frame.strip().splitlines()
    event = ""
    data = {}

    for line in lines:
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])

    return event, data


# ---------------------------------------------------------------- 1. the wire format


def test_a_token_event_is_valid_sse() -> None:
    frame = Supervisor.format_sse(StreamEvent("token", {"chunk": "ನಮಸ್ಕಾರ", "index": 1}, 1))

    assert frame == 'event: token\ndata: {"chunk": "ನಮಸ್ಕಾರ", "index": 1}\n\n'
    assert frame.endswith("\n\n")


def test_a_ping_is_an_sse_comment_not_an_event() -> None:
    frame = Supervisor.format_sse(StreamEvent("ping", {"timestamp": 123}))

    assert frame == ": ping 123\n\n"
    assert "event:" not in frame


def test_a_finish_event_carries_the_telemetry() -> None:
    frame = Supervisor.format_sse(
        StreamEvent("finish", {"done": True, "tokens_estimated": 3, "latency_ms": 12.5})
    )
    event, data = parse(frame)

    assert event == "finish"
    assert data["done"] is True
    assert data["tokens_estimated"] == 3


def test_an_error_event_is_compliant() -> None:
    event, data = parse(Supervisor.format_sse(StreamEvent("error", {"code": "UPSTREAM_ERROR", "message": "boom"})))

    assert event == "error"
    assert data["code"] == "UPSTREAM_ERROR"


def test_a_websocket_frame_is_json_with_a_type() -> None:
    frame = Supervisor.format_ws(StreamEvent("token", {"chunk": "x", "index": 1}, 1))

    payload = json.loads(frame)
    assert payload["type"] == "token"
    assert payload["chunk"] == "x"
    assert payload["index"] == 1


def test_indic_text_survives_json_encoding() -> None:
    """ensure_ascii would turn every Kannada letter into an escape a client must unescape."""
    frame = Supervisor.format_sse(StreamEvent("token", {"chunk": "ಕನ್ನಡ", "index": 1}, 1))

    assert "ಕನ್ನಡ" in frame


# ---------------------------------------------------------------- 2. the bridge


def test_tokens_stream_in_order() -> None:
    out = frames(tokens("one ", "two ", "three"))

    events = [parse(frame)[0] for frame in out]
    assert events == ["token", "token", "token", "finish"]
    chunks = [parse(frame)[1].get("chunk") for frame in out[:3]]
    assert chunks == ["one ", "two ", "three"]


def test_chips_are_injected_before_finish() -> None:
    out = frames(tokens("hello"), action_chips=[{"label": "Next", "action": "next"}])

    events = [parse(frame)[0] for frame in out]
    assert events == ["token", "chips", "finish"]
    assert parse(out[1])[1]["action_chips"] == [{"label": "Next", "action": "next"}]


def test_no_chips_event_when_there_are_none() -> None:
    out = frames(tokens("hello"))

    assert [parse(frame)[0] for frame in out] == ["token", "finish"]


def test_the_finish_event_counts_what_was_sent() -> None:
    out = frames(tokens("abcd" * 10))

    data = parse(out[-1])[1]
    assert data["done"] is True
    assert data["tokens_estimated"] == 10
    assert data["aborted_early"] is False


# ---------------------------------------------------------------- 3. the disconnect


def test_a_cancelled_client_stops_the_upstream_generator() -> None:
    """This is the line that stops the bill."""
    consumed: list[str] = []

    async def feed():
        for index in range(100):
            consumed.append(str(index))
            yield f"chunk{index} "

    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    out = frames(feed(), is_cancelled_fn=cancelled)

    assert len(consumed) < 100, "the upstream generator kept running after the client left"
    data = parse(out[-1])[1]
    assert data["aborted_early"] is True
    assert data["tokens_saved"] >= 0


def test_an_abort_still_emits_a_clean_finish() -> None:
    async def feed():
        for index in range(50):
            yield f"x{index}"

    out = frames(feed(), is_cancelled_fn=lambda: True)

    assert parse(out[-1])[0] == "finish"


# ---------------------------------------------------------------- 4. liveness


def test_a_slow_first_token_produces_a_ping() -> None:
    async def feed():
        await asyncio.sleep(0.2)
        yield "late"

    out = frames(feed(), heartbeat_interval_sec=0.05)

    assert any(frame.startswith(": ping") for frame in out)
    assert any("late" in frame for frame in out)


def test_a_fast_stream_produces_no_pings() -> None:
    out = frames(tokens("a", "b", "c"), heartbeat_interval_sec=5.0)

    assert not any(frame.startswith(": ping") for frame in out)


# ---------------------------------------------------------------- 5. failures


def test_an_upstream_error_becomes_an_error_frame() -> None:
    async def feed():
        yield "first"
        raise RuntimeError("the provider died")

    out = frames(feed())

    events = [parse(frame)[0] for frame in out]
    assert "error" in events
    assert parse([f for f in out if parse(f)[0] == "error"][0])[1]["code"] == "UPSTREAM_ERROR"


# ---------------------------------------------------------------- 6. unicode safety


def test_a_chunk_ending_in_a_virama_is_held_until_the_next() -> None:
    """A cut after a virama orphans the consonant that follows it."""
    async def feed():
        yield "ಸ್"
        yield "ಕ"

    out = frames(feed())
    chunks = [parse(frame)[1]["chunk"] for frame in out if parse(frame)[0] == "token"]

    assert chunks == ["ಸ್ಕ"]


def test_a_chunk_ending_in_a_combining_mark_is_held() -> None:
    async def feed():
        yield "ನಮಸ್ಕಾರ"
        yield "\u0cbf"  # a Telugu dependent vowel arriving on its own

    out = frames(feed())
    chunks = [parse(frame)[1]["chunk"] for frame in out if parse(frame)[0] == "token"]

    assert "".join(chunks) == "ನಮಸ್ಕಾರ\u0cbf"


def test_the_text_always_arrives_complete() -> None:
    async def feed():
        for character in "ಕನ್ನಡ ರಾಜ್ಯೋತ್ಸವ ಶುಭಾಶಯಗಳು":
            yield character

    out = frames(feed())
    text = "".join(parse(frame)[1]["chunk"] for frame in out if parse(frame)[0] == "token")

    assert text == "ಕನ್ನಡ ರಾಜ್ಯೋತ್ಸವ ಶುಭಾಶಯಗಳು"


# ---------------------------------------------------------------- 7. speed


@pytest.mark.parametrize("event_type", ["token", "chips", "finish", "error"])
def test_framing_stays_under_its_budget(event_type: str) -> None:
    event = StreamEvent(event_type, {"chunk": "ಕನ್ನಡ", "index": 1, "done": True}, 1)
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        Supervisor.format_sse(event)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
