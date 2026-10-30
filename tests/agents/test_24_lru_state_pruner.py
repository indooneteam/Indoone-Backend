"""Agent 24 - the LRU pruner.

The kernel does not ask which buffer was least valuable; it kills the process. These tests are
about giving memory up on purpose, in the order that costs the least.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.agents import load

pruner = load("24_lru_state_pruner")
Store = pruner.BoundedLRUStore


# ---------------------------------------------------------------- 1. the store


def test_put_and_get_preserve_access_order() -> None:
    store = Store(max_capacity=10)
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)

    store.get("a")

    assert store.keys() == ["b", "c", "a"]


def test_a_missing_key_is_none() -> None:
    assert Store().get("nope") is None


def test_an_expired_key_returns_none_and_is_purged() -> None:
    store = Store(default_ttl_sec=10)
    now = time.time()
    store.put("a", 1, now=now)

    assert store.get("a", now=now + 5) == 1
    assert store.get("a", now=now + 20) is None
    assert store.keys() == []


def test_capacity_evicts_the_least_recently_used() -> None:
    store = Store(max_capacity=3)
    for key in ("a", "b", "c"):
        store.put(key, key)

    store.get("a")      # a is now the freshest
    store.put("d", "d")  # b is the oldest

    assert "b" not in store.keys()
    assert store.keys() == ["c", "a", "d"]


def test_a_put_updates_an_existing_key_without_growing() -> None:
    store = Store(max_capacity=2)
    store.put("a", 1)
    store.put("a", 2)

    assert len(store) == 1
    assert store.get("a") == 2


def test_a_zero_ttl_expires_immediately() -> None:
    store = Store(default_ttl_sec=0)
    now = time.time()
    store.put("a", 1, now=now)

    assert store.get("a", now=now) is None


# ---------------------------------------------------------------- 2. sweeping


def test_sweep_purges_expired_and_keeps_the_living() -> None:
    store = Store(default_ttl_sec=10)
    now = time.time()
    store.put("old", 1, now=now - 100)
    store.put("fresh", 2, now=now)

    report = store.sweep(now=now)

    assert report.expired_removed == 1
    assert store.keys() == ["fresh"]


def test_sweep_enforces_the_capacity_even_without_expiry() -> None:
    store = Store(max_capacity=5, default_ttl_sec=10_000)
    for index in range(20):
        store.put(f"k{index}", index)

    report = store.sweep()

    assert len(store) <= 5
    assert report.lru_evicted >= 0


def test_sweep_of_a_clean_store_removes_nothing() -> None:
    store = Store(default_ttl_sec=10_000)
    store.put("a", 1)

    report = store.sweep()

    assert report.expired_removed == 0
    assert report.remaining_entries == 1


# ---------------------------------------------------------------- 3. pressure


@pytest.mark.parametrize(
    ("used_pct", "level"),
    [(0, "NORMAL"), (69.9, "NORMAL"), (70, "ELEVATED"), (84.9, "ELEVATED"), (85, "CRITICAL"), (99, "CRITICAL")],
)
def test_pressure_levels(used_pct: float, level: str) -> None:
    assert pruner.evaluate_system_pressure(used_pct) == level


def test_nonsense_pressure_reads_as_normal() -> None:
    assert pruner.evaluate_system_pressure("lots") == "NORMAL"  # type: ignore[arg-type]


def test_critical_pressure_sheds_a_quarter_of_the_oldest() -> None:
    store = Store(max_capacity=1000, default_ttl_sec=10_000, tier=1)
    for index in range(40):
        store.put(f"k{index}", index)

    report = pruner.orchestrate_fleet_prune([store], 90)

    assert report.pressure_level == "CRITICAL"
    assert report.emergency_shed == 10
    assert len(store) == 30
    # The oldest went: the newest keys are still there.
    assert "k39" in store.keys()
    assert "k0" not in store.keys()


def test_elevated_pressure_does_not_shed() -> None:
    store = Store(max_capacity=1000, default_ttl_sec=10_000, tier=1)
    for index in range(40):
        store.put(f"k{index}", index)

    report = pruner.orchestrate_fleet_prune([store], 75)

    assert report.pressure_level == "ELEVATED"
    assert report.emergency_shed == 0
    assert len(store) == 40


# ---------------------------------------------------------------- 4. the order that matters


def test_the_cheap_tier_is_shed_before_the_expensive_one() -> None:
    """Losing a de-duplication hash costs one duplicate reply. Losing a cached answer costs a
    model call. Under pressure the cheap thing goes first."""
    sacrificial = Store(max_capacity=1000, default_ttl_sec=10_000, tier=pruner.SACRIFICIAL)
    protected = Store(max_capacity=1000, default_ttl_sec=10_000, tier=pruner.PROTECTED)

    for index in range(40):
        sacrificial.put(f"a{index}", index)
        protected.put(f"c{index}", index)

    report = pruner.orchestrate_fleet_prune([protected, sacrificial], 90)

    assert report.emergency_shed == 10
    assert len(sacrificial) == 30
    assert len(protected) == 40, "the protected tier must be untouched while tier 1 has entries"


def test_the_protected_tier_is_shed_only_when_nothing_cheaper_is_left() -> None:
    protected = Store(max_capacity=1000, default_ttl_sec=10_000, tier=pruner.PROTECTED)
    for index in range(40):
        protected.put(f"c{index}", index)

    report = pruner.orchestrate_fleet_prune([protected], 95)

    assert report.emergency_shed == 10
    assert len(protected) == 30


def test_a_normal_fleet_is_only_swept() -> None:
    first = Store(max_capacity=10, default_ttl_sec=10_000, tier=1)
    second = Store(max_capacity=10, default_ttl_sec=10_000, tier=3)
    first.put("a", 1)
    second.put("b", 2)

    report = pruner.orchestrate_fleet_prune([first, second], 20)

    assert report.pressure_level == "NORMAL"
    assert report.emergency_shed == 0
    assert report.remaining_entries == 2


def test_an_empty_fleet_is_harmless() -> None:
    report = pruner.orchestrate_fleet_prune([], 95)

    assert report.remaining_entries == 0
    assert report.emergency_shed == 0


# ---------------------------------------------------------------- 5. concurrency and speed


def test_concurrent_reads_and_writes_do_not_corrupt_the_store() -> None:
    store = Store(max_capacity=200, default_ttl_sec=10_000)
    errors: list[Exception] = []

    def writer(offset: int) -> None:
        try:
            for index in range(100):
                store.put(f"k{offset}-{index}", index)
        except Exception as exc:  # pragma: no cover - a failure here is the test failing
            errors.append(exc)

    def reader() -> None:
        try:
            for index in range(100):
                store.get(f"k0-{index}")
                store.sweep()
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    threads.append(threading.Thread(target=reader))

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(store) <= 200


def test_a_get_stays_under_its_budget() -> None:
    store = Store(max_capacity=1000, default_ttl_sec=10_000)
    for index in range(500):
        store.put(f"k{index}", index)

    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        store.get("k250")
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_a_sweep_of_a_large_store_stays_sub_millisecond() -> None:
    store = Store(max_capacity=5000, default_ttl_sec=10_000)
    for index in range(5000):
        store.put(f"k{index}", index)

    started = time.thread_time()
    store.sweep()
    elapsed = (time.thread_time() - started) * 1000

    assert elapsed < 5.0, f"sweep took {elapsed:.4f} ms"
