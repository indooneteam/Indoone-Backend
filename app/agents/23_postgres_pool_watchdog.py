"""Agent 23 - the pool watchdog: watch the pool, never touch the data.

Two ways a Postgres-backed service falls over under load, and neither is the query that was
slow:

* **The pool runs out.** Connections that were never returned sit in the pool, and the next
  request waits for one that will not come. By the time the error surfaces it says
  ``PoolTimeout``, which tells nobody anything.
* **A transaction is left open.** A query that half-finished holds row locks, and every
  writer behind it waits. The database is not slow; one connection is.

This reads the pool's own counters and ``pg_stat_activity`` and says which of those is
happening. It is **read-only by construction**: there is no execute, no DDL, no connection of
its own - the counters are passed in, so the caller decides what to sample and this decides
what it means.

The status thresholds are reconciled deliberately. The spec asks for a warning above 85% and
for 18 of 20 connections to read as DEGRADED; those two statements disagree at 90%. A pool at
90% with connections still free is under pressure, not down - so DEGRADED covers 70% to just
short of full, and CRITICAL is reserved for the states where a request will actually fail:
no idle connections left, near-total saturation, or more than one stuck transaction.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

SATURATION_WARN_PCT = 85.0
DEGRADED_PCT = 70.0
CRITICAL_PCT = 95.0

IDLE_IN_TRANSACTION_SECONDS = 10.0
LONG_RUNNING_SECONDS = 15.0

_URI_PASSWORD = re.compile(r"\b((?:postgres(?:ql)?|mysql|mariadb|redis|mongodb(?:\+srv)?)://[^:\s/@]+:)([^@\s]+)(@)", re.IGNORECASE)

# Literals in a query: a password in a WHERE clause is a password in a log.
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_NUMBER_LITERAL = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_EMAIL_LITERAL = re.compile(r"\b[^@\s']+@[^@\s']+\.[A-Za-z]{2,}\b")

_MESSAGES = {
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
}


@dataclass(slots=True)
class HungQueryRecord:
    pid: int
    duration_seconds: float
    state: str
    sanitized_query_preview: str


@dataclass(slots=True)
class PoolTelemetry:
    active_connections: int = 0
    idle_connections: int = 0
    max_connections: int = 0
    saturation_pct: float = 0.0
    status: str = "HEALTHY"
    hung_queries: list[HungQueryRecord] = field(default_factory=list)
    has_starvation: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WatchdogResult:
    success: bool
    data: PoolTelemetry
    formatted_summary: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_POOL_WATCHDOG_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


# ---------------------------------------------------------------- the scrubbers


def sanitize_db_uri(uri: str) -> str:
    """A connection string without its password."""
    return _URI_PASSWORD.sub(lambda match: f"{match.group(1)}[REDACTED]{match.group(3)}", uri or "")


def sanitize_sql_query(query: str) -> str:
    """The shape of the query, with the values removed.

    ``WHERE password = 'hunter2'`` keeps its structure and loses the secret; the numbers and
    addresses are removed for the same reason.
    """
    body = query or ""
    body = _STRING_LITERAL.sub("'?'", body)
    body = _EMAIL_LITERAL.sub("[EMAIL]", body)
    body = _NUMBER_LITERAL.sub("?", body)
    body = sanitize_db_uri(body)
    return " ".join(body.split())[:300]


# ---------------------------------------------------------------- the analysis


def _hung(records: list[dict[str, Any]]) -> list[HungQueryRecord]:
    found: list[HungQueryRecord] = []

    for record in records or []:
        if not isinstance(record, dict):
            continue

        state = str(record.get("state") or "").strip().casefold()

        try:
            duration = float(record.get("duration_seconds") or record.get("duration") or 0.0)
        except (TypeError, ValueError):
            continue

        try:
            pid = int(record.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0

        if state == "idle in transaction" and duration > IDLE_IN_TRANSACTION_SECONDS:
            found.append(
                HungQueryRecord(
                    pid=pid,
                    duration_seconds=round(duration, 1),
                    state="idle in transaction",
                    sanitized_query_preview=sanitize_sql_query(str(record.get("query") or "")),
                )
            )
        elif state in {"active", "idle in transaction (aborted)"} and duration > LONG_RUNNING_SECONDS:
            found.append(
                HungQueryRecord(
                    pid=pid,
                    duration_seconds=round(duration, 1),
                    state="active",
                    sanitized_query_preview=sanitize_sql_query(str(record.get("query") or "")),
                )
            )

    found.sort(key=lambda record: record.duration_seconds, reverse=True)
    return found


def _chips(status: str) -> list[dict[str, str]]:
    if status == "CRITICAL":
        return [{"label": "Recycle Stale Connections", "action": "recycle_stale_pool"}]

    if status == "DEGRADED":
        return [{"label": "Inspect Hung Queries", "action": "inspect_hung_queries"}]

    return [{"label": "Pool Metrics", "action": "view_pool_metrics"}]


def audit_pool_state(
    active: int,
    idle: int,
    max_conns: int,
    activity_records: Optional[list[dict[str, Any]]] = None,
    *,
    language: str = "en",
) -> WatchdogResult:
    """Read the counters, decide what they mean, and say it in one line."""
    started = time.perf_counter()

    try:
        active_connections = max(0, int(active))
        idle_connections = max(0, int(idle))
        max_connections = max(0, int(max_conns))
    except (TypeError, ValueError):
        table = _MESSAGES["INVALID_ARGUMENT"]
        key = (language or "en").strip().casefold()[:2]

        return WatchdogResult(
            success=False,
            data=PoolTelemetry(),
            formatted_summary=table.get(key, table["en"]),
            error_code="INVALID_ARGUMENT",
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    saturation = round(100.0 * active_connections / max_connections, 1) if max_connections else 0.0
    hung = _hung(list(activity_records or []))

    starvation = bool(max_connections) and idle_connections == 0 and active_connections >= max_connections

    warnings: list[str] = []

    if saturation > SATURATION_WARN_PCT:
        warnings.append(f"Pool is {saturation}% saturated - above the {SATURATION_WARN_PCT:.0f}% line.")

    if starvation:
        warnings.append("No idle connections: the next request will wait for one that is not coming.")

    idle_in_transaction = [record for record in hung if record.state == "idle in transaction"]

    if idle_in_transaction:
        warnings.append(
            f"{len(idle_in_transaction)} transaction(s) left open for more than "
            f"{IDLE_IN_TRANSACTION_SECONDS:.0f}s, holding row locks."
        )

    if len(hung) > 1:
        warnings.append(f"{len(hung)} queries are stuck.")

    if starvation or saturation >= CRITICAL_PCT or len(hung) > 1:
        status = "CRITICAL"
    elif saturation >= DEGRADED_PCT or hung:
        status = "DEGRADED"
    else:
        status = "HEALTHY"

    summary_lines = [
        f"Pool: {active_connections} active, {idle_connections} idle of {max_connections} "
        f"({saturation}% saturated) - {status}"
    ]

    for record in hung[:3]:
        summary_lines.append(
            f"  pid {record.pid}: {record.state} for {record.duration_seconds:.0f}s "
            f"- {record.sanitized_query_preview}"
        )

    summary_lines.extend(f"  Warning: {message}" for message in warnings)

    return WatchdogResult(
        success=True,
        data=PoolTelemetry(
            active_connections=active_connections,
            idle_connections=idle_connections,
            max_connections=max_connections,
            saturation_pct=saturation,
            status=status,
            hung_queries=hung,
            has_starvation=starvation,
            warnings=warnings,
        ),
        formatted_summary="\n".join(summary_lines),
        action_chips=_chips(status),
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def get_tool_declarations() -> list[dict[str, Any]]:
    declaration = {
        "name": "db_pool_health",
        "description": "Read-only health of the Postgres connection pool and its transactions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    return [json.loads(json.dumps(declaration))]
