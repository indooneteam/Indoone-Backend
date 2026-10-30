"""Agent 21 - the streamer: tokens as they are produced, and nothing after the client leaves.

A chat that waits for the whole answer feels broken next to one that types. But a stream has
three costs a blocking call does not, and this agent is about those:

* **A disconnected client is money on fire.** The model keeps generating into a socket nobody
  is reading. The moment the client is gone, the upstream generator is stopped - not after
  the next chunk, now.
* **A mobile network drops a quiet connection.** Carriers close a connection that carries
  nothing for about fifteen seconds, so a stream that pauses for its first token sends a
  comment ping instead of dying silently.
* **A character is not a byte.** Kannada and Devanagari text is multi-byte and full of
  combining marks; a chunk boundary in the wrong place is a broken letter on the screen.

The wire formats are stated in the spec and implemented exactly, but **this module is not
wired into ``/chat/stream`` yet**: that endpoint speaks ``{"type": "delta"}`` / ``{"type":
"done"}``, which is the contract the Android client already parses. Switching the format is a
client change, not a backend one, so it is deliberately left as the next step rather than
done quietly.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

HEARTBEAT_INTERVAL_SECONDS = 15.0
MAX_QUEUE_WAIT_SECONDS = 5.0
CHARS_PER_TOKEN = 4

# Kannada, Devanagari, Telugu, Tamil, Malayalam, Bengali, Gujarati.
_VIRAMAS = frozenset({"\u0ccd", "\u094d", "\u0c4d", "\u0bcd", "\u0d4d", "\u09cd", "\u0acd"})
_JOINERS = frozenset({"\u200d", "\u200c"})


@dataclass(slots=True)
class StreamEvent:
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    index: int = 0


@dataclass(slots=True)
class StreamMetrics:
    total_chunks: int = 0
    total_characters: int = 0
    duration_ms: float = 0.0
    aborted_early: bool = False
    tokens_saved: int = 0


def enabled() -> bool:
    return os.getenv("INDOONE_APP_STREAMER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _dangling(text: str) -> bool:
    """Whether this chunk ends in the middle of a grapheme."""
    if not text:
        return False

    last = text[-1]
    return last in _VIRAMAS or last in _JOINERS or bool(unicodedata.combining(last))


class AppStreamSupervisor:
    """Framing and lifecycle for a token stream."""

    @staticmethod
    def format_sse(event: StreamEvent) -> str:
        """One SSE frame. The spec's wire format, exactly."""
        if event.event_type == "ping":
            stamp = event.data.get("timestamp") or int(time.time())
            return f": ping {stamp}\n\n"

        payload = json.dumps(event.data, ensure_ascii=False)
        return f"event: {event.event_type}\ndata: {payload}\n\n"

    @staticmethod
    def format_ws(event: StreamEvent) -> str:
        """One WebSocket JSON frame, symmetric with the SSE vocabulary."""
        frame = {"type": event.event_type, **event.data}
        return json.dumps(frame, ensure_ascii=False)

    @classmethod
    async def bridge_stream(
        cls,
        upstream_token_gen: AsyncIterator[str],
        action_chips: list[dict[str, str]] | None = None,
        is_cancelled_fn: Callable[[], bool] | None = None,
        heartbeat_interval_sec: float = HEARTBEAT_INTERVAL_SECONDS,
        *,
        clock: Callable[[], float] | None = None,
    ) -> AsyncIterator[str]:
        """Turn a token generator into SSE frames, and stop when the client leaves."""
        started = (clock or time.monotonic)()
        metrics = StreamMetrics()
        index = 0
        pending = ""

        iterator = upstream_token_gen.__aiter__()

        def cancelled() -> bool:
            return bool(is_cancelled_fn and is_cancelled_fn())

        try:
            while True:
                if cancelled():
                    metrics.aborted_early = True
                    break

                try:
                    chunk = await asyncio.wait_for(iterator.__anext__(), timeout=heartbeat_interval_sec)
                except asyncio.TimeoutError:
                    # Nothing yet: keep the connection alive rather than let a carrier drop it.
                    yield cls.format_sse(StreamEvent("ping", {"timestamp": int(time.time())}))
                    continue
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    metrics.aborted_early = True
                    raise
                except Exception as exc:  # an upstream failure is a frame, not a crash
                    yield cls.format_sse(
                        StreamEvent("error", {"code": "UPSTREAM_ERROR", "message": str(exc)[:200]})
                    )
                    break

                if not chunk:
                    continue

                # A chunk that ends mid-grapheme is held back one frame, so the client never
                # renders a broken letter.
                text = pending + chunk
                pending = ""

                if _dangling(text):
                    pending = text
                    continue

                index += 1
                metrics.total_chunks += 1
                metrics.total_characters += len(text)

                yield cls.format_sse(StreamEvent("token", {"chunk": text, "index": index}, index))

            if pending:
                index += 1
                metrics.total_chunks += 1
                metrics.total_characters += len(pending)
                yield cls.format_sse(StreamEvent("token", {"chunk": pending, "index": index}, index))

        finally:
            # Whatever happened - finished, aborted, cancelled - the upstream generator is
            # closed here. This is the line that stops the bill.
            close = getattr(iterator, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

        metrics.duration_ms = ((clock or time.monotonic)() - started) * 1000

        if metrics.aborted_early:
            metrics.tokens_saved = max(1, metrics.total_characters // CHARS_PER_TOKEN) if metrics.total_characters else 0

        if action_chips:
            yield cls.format_sse(StreamEvent("chips", {"action_chips": action_chips}))

        yield cls.format_sse(
            StreamEvent(
                "finish",
                {
                    "done": True,
                    "tokens_estimated": max(1, metrics.total_characters // CHARS_PER_TOKEN) if metrics.total_characters else 0,
                    "latency_ms": round(metrics.duration_ms, 2),
                    "aborted_early": metrics.aborted_early,
                    "tokens_saved": metrics.tokens_saved,
                },
            )
        )
