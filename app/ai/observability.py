"""Phase 0 observability: one span per request, per model call and per tool call.

Why this file exists
--------------------
Before we replace any layer of the backend we have to be able to see it. Today we
log, and we carry a request id, and that is all: we cannot answer what one user
message cost, which tool spent it, or where the latency went. This module records
that, and nothing else changes because of it.

What it records
---------------
- one span per HTTP request, per model call and per tool call;
- for a model call, the model, the token counts the provider reported, and the cost
  when a price is configured for that model;
- for a tool call, the tool name, whether it succeeded, whether it timed out, and how
  long it took.

What it deliberately does not record
------------------------------------
No prompts, no completions, no tool payloads, no email or repository content, and no
user identifiers. Counts, names, durations, tokens and status only. A span is a
measurement, not a copy of the conversation.

Cost is configuration, not a guess
----------------------------------
Prices change and they are provider-specific, so this module ships with an empty
price table. Set ``INDOONE_MODEL_PRICES_JSON`` to price models, for example::

    {"gemini": {"input": 0.30, "output": 2.50}, "default": {"input": 1.0, "output": 3.0}}

prices are US dollars per one million tokens. Until a model is priced its tokens are
counted and its cost is reported as ``unpriced``, which is honest about what we know.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

KINDS = ("request", "model", "tool")
MAX_RECENT_SPANS = 200


@dataclass
class Span:
    """One measured unit of work."""

    name: str
    kind: str
    duration_ms: float
    status: str = "ok"
    error: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
        }
        if self.error:
            payload["error"] = self.error
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload


@dataclass
class Trace:
    """Every span produced while serving one request."""

    request_id: str
    spans: list[Span] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "span_count": len(self.spans),
            "duration_ms": round(sum(span.duration_ms for span in self.spans), 3),
            "spans": [span.as_dict() for span in self.spans],
        }


_current: ContextVar[Trace | None] = ContextVar("indoone_trace", default=None)
_open_model_span: ContextVar["_SpanContext | None"] = ContextVar("indoone_open_model_span", default=None)
_lock = threading.Lock()

_totals: dict[str, int] = {kind: 0 for kind in KINDS}
_errors: dict[str, int] = {kind: 0 for kind in KINDS}
_duration_ms: dict[str, float] = {kind: 0.0 for kind in KINDS}
_tool_stats: dict[str, dict[str, float]] = {}
_model_stats: dict[str, dict[str, float]] = {}
_recent: deque[Span] = deque(maxlen=MAX_RECENT_SPANS)
_fallbacks = 0
_prices_cache: tuple[str, dict[str, dict[str, float]]] = ("", {})


def _prices() -> dict[str, dict[str, float]]:
    """Return the configured price table, re-read only when the setting changes."""

    global _prices_cache
    raw = os.getenv("INDOONE_MODEL_PRICES_JSON", "").strip()
    if raw == _prices_cache[0]:
        return _prices_cache[1]
    table: dict[str, dict[str, float]] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            logger.warning("ignoring INDOONE_MODEL_PRICES_JSON: it is not valid JSON")
            parsed = {}
        if isinstance(parsed, dict):
            for model, entry in parsed.items():
                if not isinstance(entry, dict):
                    continue
                try:
                    table[str(model)] = {
                        "input": float(entry.get("input", 0.0)),
                        "output": float(entry.get("output", 0.0)),
                    }
                except (TypeError, ValueError):
                    continue
    _prices_cache = (raw, table)
    return table


def price_for(model: str) -> dict[str, float] | None:
    """Return the configured price for a model, falling back to the default entry."""

    table = _prices()
    if model in table:
        return table[model]
    return table.get("default")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Return the cost in US dollars, or None when the model has no configured price."""

    price = price_for(model)
    if price is None:
        return None
    return (input_tokens / 1_000_000.0) * price["input"] + (output_tokens / 1_000_000.0) * price["output"]


def begin_request(request_id: str) -> Trace:
    """Start a trace for one request and make it current."""

    trace = Trace(request_id=request_id)
    _current.set(trace)
    return trace


def current_trace() -> Trace | None:
    return _current.get()


def clear_request() -> None:
    _current.set(None)


def record(span: Span) -> Span:
    """Record a finished span: onto the request trace, and into the aggregates."""

    with _lock:
        _totals[span.kind] = _totals.get(span.kind, 0) + 1
        _duration_ms[span.kind] = _duration_ms.get(span.kind, 0.0) + span.duration_ms
        if span.status != "ok":
            _errors[span.kind] = _errors.get(span.kind, 0) + 1
        if span.kind == "tool":
            tool_name = str(span.attributes.get("tool") or span.name)
            stats = _tool_stats.setdefault(tool_name, {"calls": 0.0, "errors": 0.0, "duration_ms": 0.0})
            stats["calls"] += 1
            stats["duration_ms"] += span.duration_ms
            if span.status != "ok":
                stats["errors"] += 1
        if span.kind == "model":
            model = str(span.attributes.get("model") or "unknown")
            stats = _model_stats.setdefault(
                model,
                {"calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "cost_usd": 0.0, "unpriced_calls": 0.0},
            )
            stats["calls"] += 1
            input_tokens = int(span.attributes.get("input_tokens") or 0)
            output_tokens = int(span.attributes.get("output_tokens") or 0)
            stats["input_tokens"] += input_tokens
            stats["output_tokens"] += output_tokens
            cost = estimate_cost(model, input_tokens, output_tokens)
            if cost is None:
                stats["unpriced_calls"] += 1
            else:
                stats["cost_usd"] += cost
        _recent.append(span)
    trace = _current.get()
    if trace is not None:
        trace.spans.append(span)
    return span


def note_fallback() -> None:
    """Count one provider fallback: the primary was at its limit and the other ran."""

    global _fallbacks
    with _lock:
        _fallbacks += 1


class _SpanContext:
    """Times a block and records it, whether it is entered with ``with`` or ``async with``."""

    def __init__(self, name: str, kind: str, attributes: dict[str, Any]) -> None:
        self.name = name
        self.kind = kind
        self.attributes = attributes
        self.failure = ""
        self._started = 0.0
        self._previous_open: "_SpanContext | None" = None

    def fail(self, reason: str) -> None:
        """Mark the block failed when it reports failure instead of raising it."""

        self.failure = reason

    def _enter(self) -> "_SpanContext":
        self._started = time.perf_counter()
        self._previous_open = _open_model_span.get()
        if self.kind == "model":
            _open_model_span.set(self)
        return self

    def _exit(self, exc: BaseException | None) -> bool:
        if self.kind == "model":
            _open_model_span.set(self._previous_open)
        duration_ms = (time.perf_counter() - self._started) * 1000.0
        status = "ok" if exc is None and not self.failure else "error"
        error = type(exc).__name__ if exc is not None else self.failure
        span = record(
            Span(
                name=self.name,
                kind=self.kind,
                duration_ms=duration_ms,
                status=status,
                error=error,
                attributes=dict(self.attributes),
            )
        )
        logger.info(
            "span",
            extra={
                "span_name": span.name,
                "span_kind": span.kind,
                "duration_ms": round(span.duration_ms, 3),
                "status": span.status,
                "span_attributes": span.attributes,
            },
        )
        return False

    def __enter__(self) -> "_SpanContext":
        return self._enter()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return self._exit(exc)

    async def __aenter__(self) -> "_SpanContext":
        return self._enter()

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        return self._exit(exc)


def track(name: str, kind: str, **attributes: Any) -> _SpanContext:
    """Measure a block.

    Use it as ``with track("name", "tool", tool=name) as span:`` or
    ``async with track(...) as span:``. Fill ``span.attributes`` while the block runs
    for values only known at the end, such as token counts, and call ``span.fail(reason)``
    when the block reports failure instead of raising it.
    """

    return _SpanContext(name, kind, attributes)


def set_model(model: str) -> None:
    """Name the model on the model span that is currently open, if there is one."""

    span = _open_model_span.get()
    if span is not None and model:
        span.attributes["model"] = str(model)


def add_model_usage(input_tokens: int, output_tokens: int) -> None:
    """Add provider-reported tokens to the open model span.

    Providers report usage per HTTP call, and the tool loop makes several calls
    inside one span, so the counts are added rather than replaced.
    """

    span = _open_model_span.get()
    if span is None:
        return
    span.attributes["input_tokens"] = int(span.attributes.get("input_tokens") or 0) + max(0, int(input_tokens))
    span.attributes["output_tokens"] = int(span.attributes.get("output_tokens") or 0) + max(0, int(output_tokens))


def measured_model_call(provider: str):
    """Measure one provider entry point as a single model span.

    Applied to the four entry points (a plain reply and a tool-calling reply, for
    each provider), so a model call is measured wherever it is made and the call
    sites themselves do not have to know about measurement.
    """

    def decorate(function):
        @functools.wraps(function)
        async def measured(*args: Any, **kwargs: Any) -> Any:
            async with track("model.call", "model", provider=provider) as span:
                return await function(*args, **kwargs)

        return measured

    return decorate


def record_request(method: str, path: str, *, status_code: int, duration_ms: float) -> Span:
    """Record one HTTP request, with the route only, never the query string."""

    return record(
        Span(
            name=f"{method.upper()} {path}",
            kind="request",
            duration_ms=duration_ms,
            status="ok" if status_code < 500 else "error",
            attributes={"status_code": status_code},
        )
    )


def snapshot() -> dict[str, Any]:
    """Return the aggregate view an operator reads, with no user data in it."""

    with _lock:
        totals = dict(_totals)
        errors = dict(_errors)
        durations = dict(_duration_ms)
        tool_stats = {name: dict(stats) for name, stats in _tool_stats.items()}
        model_stats = {name: dict(stats) for name, stats in _model_stats.items()}
        recent = [span.as_dict() for span in _recent]
        fallbacks = _fallbacks
    averages = {
        kind: round(durations.get(kind, 0.0) / totals[kind], 3) if totals.get(kind) else 0.0
        for kind in KINDS
    }
    return {
        "spans": {"total": totals, "errors": errors, "average_duration_ms": averages},
        "tools": tool_stats,
        "models": model_stats,
        "provider_fallbacks": fallbacks,
        "recent_spans": recent,
        "note": "counts, names, durations and tokens only; no prompts, payloads or user ids",
    }


def reset() -> None:
    """Clear every aggregate. For tests, and for an operator who wants a clean window."""

    global _fallbacks, _prices_cache
    with _lock:
        for kind in KINDS:
            _totals[kind] = 0
            _errors[kind] = 0
            _duration_ms[kind] = 0.0
        _tool_stats.clear()
        _model_stats.clear()
        _recent.clear()
        _fallbacks = 0
    _prices_cache = ("", {})
    _current.set(None)
