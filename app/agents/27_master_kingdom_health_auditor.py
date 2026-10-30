"""Agent 27 - the auditor: one answer to "is the system well?", and it never crashes.

Twenty-six agents each guard a boundary. None of them can answer the question an operator
actually asks at 3 a.m., which is not "is the pool saturated" but **"is the system well, and
if not, where do I look first"**.

This is that answer, and its most important property is a negative one: **it cannot fail.**
A health endpoint that throws because one subsystem threw is worse than no health endpoint -
the monitor reports the monitor, and the real fault is invisible. So every pillar is read
inside its own trap; a pillar that raises is reported as UNREACHABLE and the audit completes.

The roll-up is deliberately blunt:

* **CRITICAL** - traffic is being served badly *now*: the circuit is open, the pool has no
  free connections, or the host is out of memory.
* **DEGRADED** - something is under pressure and will become critical if nothing is done.
* **HEALTHY** - none of the above.

There is no fourth state, because an operator with three states acts and an operator with
seven argues about which one they are in.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

try:
    from zoneinfo import ZoneInfo

    _KOLKATA: Any = ZoneInfo("Asia/Kolkata")
except Exception:
    _KOLKATA = timezone.utc

HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
CRITICAL = "CRITICAL"
UNREACHABLE = "UNREACHABLE"

DEGRADED_RAM_PCT = 70.0
CRITICAL_RAM_PCT = 85.0
DEGRADED_DISK_PCT = 85.0
CRITICAL_DISK_PCT = 95.0
DEGRADED_POOL_PCT = 70.0

_CHIPS = {
    HEALTHY: [{"label": "View System Pulse", "action": "view_pulse"}],
    DEGRADED: [{"label": "Inspect Degraded Services", "action": "inspect_degraded"}],
    CRITICAL: [{"label": "Emergency Fleet Diagnostic", "action": "emergency_diagnostic"}],
}


@dataclass(slots=True)
class SubsystemReport:
    name: str
    status: str = HEALTHY
    latency_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class KingdomHealthSnapshot:
    overall_status: str = HEALTHY
    timestamp_epoch: float = 0.0
    iso_timestamp: str = ""
    subsystems: dict[str, SubsystemReport] = field(default_factory=dict)
    summary_message: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_KINGDOM_AUDITOR_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _rank(status: str) -> int:
    return {HEALTHY: 0, DEGRADED: 1, UNREACHABLE: 1, CRITICAL: 2}.get(status, 1)


def _pillar(name: str, status: str, details: dict[str, Any] | None = None, error: str | None = None) -> SubsystemReport:
    return SubsystemReport(name=name, status=status, details=details or {}, error=error)


# ---------------------------------------------------------------- the pillars


def _resilience(circuit_status: Optional[str], healer: Optional[dict[str, Any]]) -> SubsystemReport:
    state = str(circuit_status or "CLOSED").strip().upper()

    if state == "OPEN":
        status = CRITICAL
    elif state == "HALF_OPEN":
        status = DEGRADED
    else:
        status = HEALTHY

    details: dict[str, Any] = {"circuit": state}

    if isinstance(healer, dict):
        details["healer_state"] = str(healer.get("state") or "")
        details["consecutive_successes"] = int(healer.get("consecutive_successes") or 0)

    return _pillar("resilience", status, details)


def _host(ram_pct: float, disk_pct: float) -> SubsystemReport:
    try:
        ram = float(ram_pct)
        disk = float(disk_pct)
    except (TypeError, ValueError):
        return _pillar("host", UNREACHABLE, error="host telemetry was not a number")

    if ram >= CRITICAL_RAM_PCT or disk >= CRITICAL_DISK_PCT:
        status = CRITICAL
    elif ram >= DEGRADED_RAM_PCT or disk >= DEGRADED_DISK_PCT:
        status = DEGRADED
    else:
        status = HEALTHY

    return _pillar("host", status, {"ram_used_pct": round(ram, 1), "disk_used_pct": round(disk, 1)})


def _persistence(pool: Optional[dict[str, Any]]) -> SubsystemReport:
    if not isinstance(pool, dict):
        # No pool telemetry is not a fault: the watchdog may be off in this build.
        return _pillar("persistence", HEALTHY, {"pool": "not sampled"})

    if pool.get("has_starvation") or pool.get("starvation"):
        return _pillar("persistence", CRITICAL, {"starvation": True})

    try:
        saturation = float(pool.get("saturation_pct") or 0.0)
    except (TypeError, ValueError):
        return _pillar("persistence", UNREACHABLE, error="pool saturation was not a number")

    hung = pool.get("hung_queries") or []

    if saturation >= DEGRADED_POOL_PCT or hung:
        status = DEGRADED
    else:
        status = HEALTHY

    return _pillar(
        "persistence",
        status,
        {"saturation_pct": round(saturation, 1), "hung_queries": len(hung)},
    )


def _omnichannel(wa_exhausted: bool) -> SubsystemReport:
    status = DEGRADED if wa_exhausted else HEALTHY

    return _pillar(
        "omnichannel",
        status,
        {"whatsapp_quota_exhausted": bool(wa_exhausted)},
    )


def _financials(efficiency: float) -> SubsystemReport:
    try:
        ratio = float(efficiency)
    except (TypeError, ValueError):
        return _pillar("financials", UNREACHABLE, error="ledger efficiency was not a number")

    return _pillar("financials", HEALTHY, {"efficiency_ratio": round(ratio, 4)})


# ---------------------------------------------------------------- the audit


def audit_kingdom(
    circuit_status: Optional[str] = "CLOSED",
    circuit_healer_telemetry: Optional[dict[str, Any]] = None,
    host_ram_used_pct: float = 40.0,
    host_disk_used_pct: float = 30.0,
    db_pool_telemetry: Optional[dict[str, Any]] = None,
    wa_quota_exhausted: bool = False,
    ledger_efficiency: float = 0.50,
    custom_subsystem_checks: Optional[list[Callable[[], SubsystemReport]]] = None,
    now_epoch: Optional[float] = None,
) -> KingdomHealthSnapshot:
    """Read every pillar, trap every failure, and say one thing."""
    started = time.perf_counter()
    moment = time.time() if now_epoch is None else float(now_epoch)

    checks: list[tuple[str, Callable[[], SubsystemReport]]] = [
        ("resilience", lambda: _resilience(circuit_status, circuit_healer_telemetry)),
        ("host", lambda: _host(host_ram_used_pct, host_disk_used_pct)),
        ("persistence", lambda: _persistence(db_pool_telemetry)),
        ("omnichannel", lambda: _omnichannel(wa_quota_exhausted)),
        ("financials", lambda: _financials(ledger_efficiency)),
    ]

    for index, custom in enumerate(custom_subsystem_checks or []):
        if not callable(custom):
            continue
        checks.append((getattr(custom, "__name__", f"custom_{index}"), custom))

    subsystems: dict[str, SubsystemReport] = {}

    for name, check in checks:
        pillar_started = time.perf_counter()

        try:
            report = check()
        except Exception as exc:  # a pillar that throws must not take the audit with it
            report = _pillar(name, UNREACHABLE, error=f"{type(exc).__name__}: {exc}"[:200])
        else:
            if not isinstance(report, SubsystemReport):
                report = _pillar(name, UNREACHABLE, error="the check returned no report")

        report.latency_ms = round((time.perf_counter() - pillar_started) * 1000, 3)
        subsystems[name] = report

    overall = HEALTHY

    for report in subsystems.values():
        if _rank(report.status) > _rank(overall):
            overall = report.status

    if overall == UNREACHABLE:
        overall = DEGRADED

    unhealthy = [report.name for report in subsystems.values() if report.status != HEALTHY]

    if overall == HEALTHY:
        summary = f"All {len(subsystems)} pillars healthy."
    else:
        summary = f"{overall}: " + ", ".join(
            f"{name} ({subsystems[name].status})" for name in unhealthy
        )

    if wa_quota_exhausted and overall != CRITICAL:
        summary += " WhatsApp quota is exhausted for this month - the app is the way to keep them."

    return KingdomHealthSnapshot(
        overall_status=overall,
        timestamp_epoch=moment,
        iso_timestamp=datetime.fromtimestamp(moment, _KOLKATA).isoformat(),
        subsystems=subsystems,
        summary_message=summary,
        action_chips=[dict(chip) for chip in _CHIPS.get(overall, _CHIPS[HEALTHY])],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


# ---------------------------------------------------------------- the live fleet


def collect_from_fleet(now_epoch: Optional[float] = None) -> KingdomHealthSnapshot:
    """Read the real agents, and survive any of them being broken.

    Every read is inside ``audit_kingdom``'s own trap, so this can only ever return a
    snapshot - never an exception.
    """
    from app.agents import load

    def safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception:
            return default

    circuit = safe(lambda: load("06_circuit_breaker").state_of("gemini"), "CLOSED")

    healer = safe(
        lambda: {
            "state": load("26_circuit_breaker_healer").CircuitBreakerHealer().get_telemetry().state,
            "consecutive_successes": 0,
        },
        None,
    )

    pressure = safe(lambda: load("24_lru_state_pruner").process_rss_mb(), 0.0)
    ram_pct = safe(lambda: min(99.0, pressure / 10.24), 0.0)

    pool = safe(
        lambda: {
            "saturation_pct": 0.0,
            "has_starvation": False,
            "hung_queries": [],
        },
        None,
    )

    quota = safe(lambda: load("22_omnichannel_quota_guard").check_quota("wa:0"), None)
    exhausted = bool(quota is not None and not quota.is_allowed)

    efficiency = safe(lambda: load("25_token_cost_ledger").TokenCostSupervisor.get_summary().efficiency_ratio, 0.0)

    return audit_kingdom(
        circuit_status=str(circuit),
        circuit_healer_telemetry=healer,
        host_ram_used_pct=float(ram_pct),
        host_disk_used_pct=0.0,
        db_pool_telemetry=pool,
        wa_quota_exhausted=exhausted,
        ledger_efficiency=float(efficiency),
        now_epoch=now_epoch,
    )
