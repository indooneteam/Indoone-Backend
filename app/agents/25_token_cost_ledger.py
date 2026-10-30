"""Agent 25 - the ledger: what the system spent, and what it did not have to.

Four agents exist to avoid spending money - the 0-token intent path, the answer cache, the
context condenser, and the stream abort. Without a ledger, none of that shows up anywhere: the
bill is the same shape either way, and the only evidence that any of it works is a story
somebody tells in a meeting.

So every model call is costed against the rate card, and every avoided call is costed against
the same card. Then one number says whether the optimisations are real:

    efficiency = saved / (spent + saved)

A tenth of a cent is the unit here, so the arithmetic is done in **micro-dollars as integers**.
Floating point accumulates a rounding error per call, and a ledger that drifts is worse than
no ledger - it is a confident wrong number.

This is not the per-user quota (agent 09 owns that). That answers "may this person spend
more?"; this answers "what did we spend, on what, and what did we avoid?"
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

# Dollars per million tokens, as (input, output, cached).
_PRICING: dict[str, tuple[float, float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30, 0.01875),
    "gemini-1.5-pro": (1.25, 5.00, 0.3125),
    "azure-gpt-4o-mini": (0.15, 0.60, 0.075),
}

# Unknown models are priced at the most expensive card we know, so a new model cannot
# silently under-report. Over-reporting is recoverable; under-reporting is a surprise invoice.
_FALLBACK = _PRICING["gemini-1.5-pro"]

MAX_RECORDS = 10_000

FAST_PATH = "FAST_PATH"
CACHE_HIT = "CACHE_HIT"
CONDENSER = "CONDENSER"
STREAM_ABORT = "STREAM_ABORT"

_MECHANISM_ALIASES = {
    "FAST_PATH_INTENT": FAST_PATH,
    "FAST_PATH": FAST_PATH,
    "TREASURY_CACHE_HIT": CACHE_HIT,
    "CACHE_HIT": CACHE_HIT,
    "CONTEXT_COMPRESSION": CONDENSER,
    "CONDENSER": CONDENSER,
    "STREAM_ABORT": STREAM_ABORT,
    "ABORT": STREAM_ABORT,
}


@dataclass(slots=True)
class TokenUsageRecord:
    channel: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    timestamp_epoch: float


@dataclass(slots=True)
class SavingsRecord:
    mechanism: str
    channel: str
    estimated_tokens_saved: int
    cost_saved_usd: float
    timestamp_epoch: float


@dataclass(slots=True)
class LedgerSummary:
    total_spent_usd: float = 0.0
    total_saved_usd: float = 0.0
    efficiency_ratio: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens_saved: int = 0
    by_channel_spend: dict[str, float] = field(default_factory=dict)
    by_model_spend: dict[str, float] = field(default_factory=dict)
    by_mechanism_savings: dict[str, float] = field(default_factory=dict)
    telemetry_micros: float = 0.0


_usage: Deque[TokenUsageRecord] = deque(maxlen=MAX_RECORDS)
_savings: Deque[SavingsRecord] = deque(maxlen=MAX_RECORDS)
_lock = threading.RLock()


def enabled() -> bool:
    return os.getenv("INDOONE_COST_LEDGER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _micro(dollars: float) -> int:
    return int(round(dollars * 1_000_000))


def _dollars(micro: int) -> float:
    return round(micro / 1_000_000, 6)


def rates_for(model: str) -> tuple[float, float, float]:
    """The rate card for a model, or the highest one we know."""
    key = (model or "").strip().casefold()

    if key in _PRICING:
        return _PRICING[key]

    # A dated or suffixed identifier is still the family it names.
    for known, rates in _PRICING.items():
        if key.startswith(known) or known in key:
            return rates

    return _FALLBACK


class TokenCostSupervisor:
    """Costing and attribution, in integers."""

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        input_rate, output_rate, cached_rate = rates_for(model)

        try:
            prompt = max(0, int(input_tokens))
            completion = max(0, int(output_tokens))
            cached = max(0, int(cached_tokens))
        except (TypeError, ValueError):
            return 0.0

        micro = (
            _micro(prompt / 1_000_000 * input_rate)
            + _micro(completion / 1_000_000 * output_rate)
            + _micro(cached / 1_000_000 * cached_rate)
        )

        return _dollars(micro)

    @classmethod
    def record_usage(
        cls,
        channel: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        *,
        now: float | None = None,
    ) -> TokenUsageRecord:
        record = TokenUsageRecord(
            channel=(channel or "app").strip().casefold(),
            model=(model or "unknown").strip(),
            input_tokens=max(0, int(input_tokens or 0)),
            output_tokens=max(0, int(output_tokens or 0)),
            cached_tokens=max(0, int(cached_tokens or 0)),
            cost_usd=cls.calculate_cost(model, input_tokens, output_tokens, cached_tokens),
            timestamp_epoch=time.time() if now is None else float(now),
        )

        with _lock:
            _usage.append(record)

        return record

    @classmethod
    def record_savings(
        cls,
        mechanism: str,
        channel: str,
        tokens_saved: int,
        reference_model: str = "gemini-1.5-flash",
        *,
        now: float | None = None,
    ) -> SavingsRecord:
        """What the avoided call would have cost, at the model it would have gone to."""
        name = _MECHANISM_ALIASES.get((mechanism or "").strip().upper(), (mechanism or "").strip().upper())
        tokens = max(0, int(tokens_saved or 0))

        record = SavingsRecord(
            mechanism=name,
            channel=(channel or "app").strip().casefold(),
            estimated_tokens_saved=tokens,
            cost_saved_usd=cls.calculate_cost(reference_model, 0, tokens),
            timestamp_epoch=time.time() if now is None else float(now),
        )

        with _lock:
            _savings.append(record)

        return record

    @classmethod
    def get_summary(cls, since_epoch: Optional[float] = None, *, now: float | None = None) -> LedgerSummary:
        started = time.perf_counter()
        moment = time.time() if now is None else float(now)

        with _lock:
            usage = [record for record in _usage if since_epoch is None or record.timestamp_epoch >= since_epoch]
            savings = [
                record for record in _savings if since_epoch is None or record.timestamp_epoch >= since_epoch
            ]

        spent_micro = sum(_micro(record.cost_usd) for record in usage)
        saved_micro = sum(_micro(record.cost_saved_usd) for record in savings)

        by_channel: dict[str, int] = {}
        by_model: dict[str, int] = {}

        for record in usage:
            by_channel[record.channel] = by_channel.get(record.channel, 0) + _micro(record.cost_usd)
            by_model[record.model] = by_model.get(record.model, 0) + _micro(record.cost_usd)

        by_mechanism: dict[str, int] = {}

        for record in savings:
            by_mechanism[record.mechanism] = by_mechanism.get(record.mechanism, 0) + _micro(record.cost_saved_usd)

        hypothetical = spent_micro + saved_micro

        return LedgerSummary(
            total_spent_usd=_dollars(spent_micro),
            total_saved_usd=_dollars(saved_micro),
            efficiency_ratio=round(saved_micro / hypothetical, 4) if hypothetical else 0.0,
            total_input_tokens=sum(record.input_tokens for record in usage),
            total_output_tokens=sum(record.output_tokens for record in usage),
            total_cached_tokens=sum(record.cached_tokens for record in usage),
            total_tokens_saved=sum(record.estimated_tokens_saved for record in savings),
            by_channel_spend={name: _dollars(micro) for name, micro in sorted(by_channel.items())},
            by_model_spend={name: _dollars(micro) for name, micro in sorted(by_model.items())},
            by_mechanism_savings={name: _dollars(micro) for name, micro in sorted(by_mechanism.items())},
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    @classmethod
    def reset_state_for_tests(cls) -> None:
        with _lock:
            _usage.clear()
            _savings.clear()


def action_chips(summary: LedgerSummary) -> list[dict[str, str]]:
    """The interesting chip is the savings report, and only when there is something to show."""
    if summary.total_saved_usd > summary.total_spent_usd and summary.total_saved_usd > 0:
        return [{"label": "View Cost Savings Report", "action": "view_savings_report"}]

    return [{"label": "View Token Spend", "action": "view_token_spend"}]
