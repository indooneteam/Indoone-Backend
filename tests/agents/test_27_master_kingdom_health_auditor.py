"""Agent 27 - the kingdom auditor.

Its most important property is negative: it cannot fail. A health endpoint that throws because
one subsystem threw reports the monitor instead of the fault, and the real problem goes unseen.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.agents import load

auditor = load("27_master_kingdom_health_auditor")

KOLKATA = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------- 1. the healthy case


def test_everything_green_rolls_up_to_healthy() -> None:
    snapshot = auditor.audit_kingdom()

    assert snapshot.overall_status == "HEALTHY"
    assert snapshot.summary_message.startswith("All 5 pillars healthy")
    assert all(report.status == "HEALTHY" for report in snapshot.subsystems.values())


def test_the_five_pillars_are_all_present() -> None:
    snapshot = auditor.audit_kingdom()

    assert set(snapshot.subsystems) == {"resilience", "host", "persistence", "omnichannel", "financials"}


def test_a_healthy_snapshot_carries_the_pulse_chip() -> None:
    assert [chip["action"] for chip in auditor.audit_kingdom().action_chips] == ["view_pulse"]


# ---------------------------------------------------------------- 2. the critical cases


def test_an_open_circuit_is_critical_immediately() -> None:
    snapshot = auditor.audit_kingdom(circuit_status="OPEN")

    assert snapshot.overall_status == "CRITICAL"
    assert snapshot.subsystems["resilience"].status == "CRITICAL"
    assert [chip["action"] for chip in snapshot.action_chips] == ["emergency_diagnostic"]


def test_a_half_open_circuit_is_degraded() -> None:
    snapshot = auditor.audit_kingdom(circuit_status="HALF_OPEN")

    assert snapshot.overall_status == "DEGRADED"


def test_pool_starvation_is_critical() -> None:
    snapshot = auditor.audit_kingdom(db_pool_telemetry={"has_starvation": True, "saturation_pct": 100.0})

    assert snapshot.overall_status == "CRITICAL"
    assert snapshot.subsystems["persistence"].status == "CRITICAL"


def test_host_memory_at_eighty_six_percent_is_critical() -> None:
    assert auditor.audit_kingdom(host_ram_used_pct=86.0).overall_status == "CRITICAL"


# ---------------------------------------------------------------- 3. the degraded cases


def test_host_memory_at_seventy_five_percent_is_degraded() -> None:
    snapshot = auditor.audit_kingdom(host_ram_used_pct=75.0)

    assert snapshot.overall_status == "DEGRADED"
    assert snapshot.subsystems["host"].status == "DEGRADED"
    assert [chip["action"] for chip in snapshot.action_chips] == ["inspect_degraded"]


def test_an_exhausted_whatsapp_quota_is_degraded_with_a_recommendation() -> None:
    snapshot = auditor.audit_kingdom(wa_quota_exhausted=True)

    assert snapshot.overall_status == "DEGRADED"
    assert "app is the way" in snapshot.summary_message


def test_a_busy_pool_is_degraded() -> None:
    snapshot = auditor.audit_kingdom(db_pool_telemetry={"saturation_pct": 78.0, "hung_queries": []})

    assert snapshot.overall_status == "DEGRADED"


def test_a_hung_query_is_degraded_even_at_low_saturation() -> None:
    snapshot = auditor.audit_kingdom(
        db_pool_telemetry={"saturation_pct": 20.0, "hung_queries": [{"pid": 1}]}
    )

    assert snapshot.subsystems["persistence"].status == "DEGRADED"


def test_the_summary_names_what_is_wrong() -> None:
    snapshot = auditor.audit_kingdom(circuit_status="OPEN", wa_quota_exhausted=True)

    assert "resilience" in snapshot.summary_message
    assert "CRITICAL" in snapshot.summary_message


# ---------------------------------------------------------------- 4. the zero-cascade guarantee


def test_a_throwing_subsystem_is_trapped_and_the_audit_completes() -> None:
    def broken() -> object:
        raise RuntimeError("the database is on fire")

    snapshot = auditor.audit_kingdom(custom_subsystem_checks=[broken])

    assert snapshot.overall_status == "DEGRADED"
    assert snapshot.subsystems["broken"].status == "UNREACHABLE"
    assert "the database is on fire" in snapshot.subsystems["broken"].error
    # The other pillars were still evaluated.
    assert snapshot.subsystems["host"].status == "HEALTHY"


def test_a_check_returning_nothing_is_unreachable_not_a_crash() -> None:
    snapshot = auditor.audit_kingdom(custom_subsystem_checks=[lambda: None])

    assert list(snapshot.subsystems.values())[-1].status == "UNREACHABLE"


def test_a_non_callable_in_the_list_is_skipped() -> None:
    snapshot = auditor.audit_kingdom(custom_subsystem_checks=["not a function"])  # type: ignore[list-item]

    assert set(snapshot.subsystems) == {"resilience", "host", "persistence", "omnichannel", "financials"}


def test_nonsense_telemetry_is_reported_rather_than_raised() -> None:
    snapshot = auditor.audit_kingdom(host_ram_used_pct="lots")  # type: ignore[arg-type]

    assert snapshot.subsystems["host"].status == "UNREACHABLE"
    assert snapshot.overall_status == "DEGRADED"


def test_a_missing_pool_sample_is_not_a_fault() -> None:
    """The watchdog may be off in this build; that is not an outage."""
    snapshot = auditor.audit_kingdom(db_pool_telemetry=None)

    assert snapshot.subsystems["persistence"].status == "HEALTHY"
    assert snapshot.subsystems["persistence"].details["pool"] == "not sampled"


# ---------------------------------------------------------------- 5. the timestamp


def test_the_timestamp_is_kolkata_time() -> None:
    snapshot = auditor.audit_kingdom(now_epoch=1_700_000_000.0)
    parsed = datetime.fromisoformat(snapshot.iso_timestamp)

    assert parsed.utcoffset() == KOLKATA.utcoffset(parsed)
    assert snapshot.timestamp_epoch == 1_700_000_000.0


# ---------------------------------------------------------------- 6. the live fleet


def test_the_live_fleet_read_returns_a_snapshot() -> None:
    snapshot = auditor.collect_from_fleet()

    assert snapshot.overall_status in {"HEALTHY", "DEGRADED", "CRITICAL"}
    assert snapshot.subsystems


def test_each_pillar_records_its_own_latency() -> None:
    snapshot = auditor.audit_kingdom()

    for report in snapshot.subsystems.values():
        assert report.latency_ms >= 0.0


# ---------------------------------------------------------------- 7. speed


def test_the_roll_up_stays_under_its_budget() -> None:
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        auditor.audit_kingdom()
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


# ---------------------------------------------------------------- 8. the endpoint


def test_the_endpoint_answers_with_a_snapshot(monkeypatch) -> None:
    """A new path, so the deploy script's /health is untouched."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)

    response = TestClient(app).get("/health/kingdom")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"HEALTHY", "DEGRADED", "CRITICAL"}
    assert "subsystems" in body
    assert "summary" in body


def test_the_plain_health_endpoint_is_unchanged(monkeypatch) -> None:
    """/health is the deployment contract; adding a view must not alter it."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)

    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_the_kingdom_endpoint_is_reachable_without_a_token(monkeypatch) -> None:
    """A monitor that needs an app-user token cannot report that the app is down."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")

    response = TestClient(app).get("/health/kingdom")

    assert response.status_code == 200
    assert response.json()["status"] in {"HEALTHY", "DEGRADED", "CRITICAL"}


def test_a_chat_request_is_still_refused_without_a_token(monkeypatch) -> None:
    """The exemption is a path, not a hole in the auth."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_AUTH_REQUIRED", "true")

    response = TestClient(app).post("/api/chat", json={"message": "hello"})

    assert response.status_code == 401
