"""Agent 24 - the pruner: keep the process alive, and decide what to forget.

Every in-memory buffer this system grew - the treasury cache, three de-duplication windows,
the session buffer - is a promise that the RAM holds. On a 1 GB Lightsail box, a promise
nobody enforces is a SIGKILL: the kernel does not ask which buffer was least valuable, it
picks the whole process.

So the memory has to be given up on purpose, in an order that is decided here:

* **Expired first.** A record past its TTL is already worthless; removing it costs nothing.
* **Then the least recently used.** Capacity is a ceiling, not a target.
* **Then, under real pressure, the cheapest value.** De-duplication hashes are worth seconds
  of work; a cached answer is worth a model call. So tier 1 goes before tier 3, and tier 3 is
  only touched when shedding the cheaper tiers was not enough.

The order is the whole point: shedding the treasury cache first would free the same bytes and
throw away the most expensive thing in the system.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

NORMAL_BELOW_PCT = 70.0
CRITICAL_AT_PCT = 85.0
EMERGENCY_SHED_FRACTION = 0.25

SACRIFICIAL = 1
EPHEMERAL = 2
PROTECTED = 3


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: Any
    expires_at_epoch: float
    last_accessed_epoch: float
    tier: int = EPHEMERAL


@dataclass(slots=True)
class PruneReport:
    expired_removed: int = 0
    lru_evicted: int = 0
    emergency_shed: int = 0
    remaining_entries: int = 0
    pressure_level: str = "NORMAL"
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_LRU_PRUNER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def evaluate_system_pressure(used_pct: float) -> str:
    """Three levels, because two would either panic early or panic late."""
    try:
        used = float(used_pct)
    except (TypeError, ValueError):
        return "NORMAL"

    if used >= CRITICAL_AT_PCT:
        return "CRITICAL"
    if used >= NORMAL_BELOW_PCT:
        return "ELEVATED"
    return "NORMAL"


class BoundedLRUStore:
    """A dict with a ceiling, a clock, and a lock."""

    def __init__(self, max_capacity: int = 1024, default_ttl_sec: float = 900.0, tier: int = EPHEMERAL) -> None:
        self.max_capacity = max(1, int(max_capacity))
        self.default_ttl_sec = max(0.0, float(default_ttl_sec))
        self.tier = int(tier)
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def put(self, key: str, value: Any, ttl_sec: Optional[float] = None, *, now: float | None = None) -> None:
        moment = time.time() if now is None else float(now)
        ttl = self.default_ttl_sec if ttl_sec is None else max(0.0, float(ttl_sec))

        with self._lock:
            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                expires_at_epoch=moment + ttl,
                last_accessed_epoch=moment,
                tier=self.tier,
            )
            self._entries.move_to_end(key)

            # A ceiling, not a target: the oldest goes the moment it is crossed.
            while len(self._entries) > self.max_capacity:
                self._entries.popitem(last=False)

    def get(self, key: str, *, now: float | None = None) -> Any:
        moment = time.time() if now is None else float(now)

        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                return None

            if entry.expires_at_epoch <= moment:
                self._entries.pop(key, None)
                return None

            entry.last_accessed_epoch = moment
            self._entries.move_to_end(key)
            return entry.value

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def evict_expired(self, *, now: float | None = None) -> int:
        moment = time.time() if now is None else float(now)

        with self._lock:
            stale = [key for key, entry in self._entries.items() if entry.expires_at_epoch <= moment]

            for key in stale:
                self._entries.pop(key, None)

            return len(stale)

    def evict_lru(self, count: int) -> int:
        with self._lock:
            removed = 0

            while removed < count and self._entries:
                self._entries.popitem(last=False)
                removed += 1

            return removed

    def shed_fraction(self, fraction: float) -> int:
        """Give up the oldest slice of this store."""
        with self._lock:
            target = int(len(self._entries) * max(0.0, min(1.0, fraction)))
            return self.evict_lru(target)

    def sweep(self, force_shed_pct: float = 0.0, *, now: float | None = None) -> PruneReport:
        """Expire, then cap, then - only if asked - shed."""
        moment = time.time() if now is None else float(now)
        started = time.perf_counter()

        with self._lock:
            expired = self.evict_expired(now=moment)

            over_capacity = max(0, len(self._entries) - self.max_capacity)
            evicted = self.evict_lru(over_capacity) if over_capacity else 0

            shed = 0

            if force_shed_pct > 0:
                shed = self.shed_fraction(force_shed_pct / 100.0 if force_shed_pct > 1 else force_shed_pct)

            remaining = len(self._entries)

        return PruneReport(
            expired_removed=expired,
            lru_evicted=evicted,
            emergency_shed=shed,
            remaining_entries=remaining,
            pressure_level=evaluate_system_pressure(0.0),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )


def orchestrate_fleet_prune(stores: list[BoundedLRUStore], host_ram_used_pct: float) -> PruneReport:
    """Sweep every store, and shed in the order that costs the least.

    Tier 1 is a de-duplication hash: losing it means one duplicate reply. Tier 3 is a cached
    answer: losing it means a model call. Under pressure, the cheap thing goes first - and
    the expensive thing is only touched when the cheap tiers were not enough.
    """
    started = time.perf_counter()
    pressure = evaluate_system_pressure(host_ram_used_pct)

    ordered = sorted(stores or [], key=lambda store: store.tier)

    expired = 0
    evicted = 0

    for store in ordered:
        report = store.sweep()
        expired += report.expired_removed
        evicted += report.lru_evicted

    shed = 0

    if pressure == "CRITICAL":
        for store in ordered:
            if store.tier >= PROTECTED:
                continue
            shed += store.shed_fraction(EMERGENCY_SHED_FRACTION)

        # Still holding too much after the cheap tiers are gone: then the protected cache
        # gives up a quarter too, because a killed process loses all of it.
        if sum(len(store) for store in ordered) > 0 and not shed:
            for store in ordered:
                shed += store.shed_fraction(EMERGENCY_SHED_FRACTION)

    remaining = sum(len(store) for store in ordered)

    return PruneReport(
        expired_removed=expired,
        lru_evicted=evicted,
        emergency_shed=shed,
        remaining_entries=remaining,
        pressure_level=pressure,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def process_rss_mb() -> float:
    """This process's resident memory, from the OS - no dependency, no shell."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Linux reports kilobytes, macOS bytes.
        return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def memory_pressure_now(total_mb: float = 0.0) -> str:
    """Pressure from the host's own figures when a caller has them, else from this process."""
    if total_mb and total_mb > 0:
        return evaluate_system_pressure(100.0 * process_rss_mb() / total_mb)

    return "NORMAL"
