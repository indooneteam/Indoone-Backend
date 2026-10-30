"""Agent 23 - the pool watchdog.

Two failures look the same from outside and are not: the pool being full, and one connection
holding a lock. These tests separate them, and check that neither path can leak a password.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

watchdog = load("23_postgres_pool_watchdog")


def audit(*args, **kwargs):
    return watchdog.audit_pool_state(*args, **kwargs)


# ---------------------------------------------------------------- 1. status


def test_a_normal_pool_is_healthy() -> None:
    result = audit(5, 15, 20)

    assert result.success is True
    assert result.data.status == "HEALTHY"
    assert result.data.saturation_pct == 25.0
    assert result.data.has_starvation is False
    assert result.data.warnings == []


def test_a_busy_pool_is_degraded_with_a_warning() -> None:
    result = audit(18, 2, 20)

    assert result.data.status == "DEGRADED"
    assert result.data.saturation_pct == 90.0
    assert any("saturated" in message for message in result.data.warnings)
    assert result.data.has_starvation is False


def test_a_full_pool_is_critical_and_starved() -> None:
    result = audit(20, 0, 20)

    assert result.data.status == "CRITICAL"
    assert result.data.has_starvation is True
    assert any("No idle connections" in message for message in result.data.warnings)


def test_the_thresholds_are_ordered_the_way_the_summary_reads() -> None:
    """70% is where it starts to hurt; full is where a request actually fails."""
    assert audit(13, 7, 20).data.status == "HEALTHY"   # 65%
    assert audit(14, 6, 20).data.status == "DEGRADED"  # 70%
    assert audit(18, 2, 20).data.status == "DEGRADED"  # 90%, still a connection free
    assert audit(20, 0, 20).data.status == "CRITICAL"  # none free


def test_a_zero_maximum_does_not_divide_by_zero() -> None:
    result = audit(0, 0, 0)

    assert result.success is True
    assert result.data.saturation_pct == 0.0
    assert result.data.status == "HEALTHY"


# ---------------------------------------------------------------- 2. the hung queries


def test_an_idle_transaction_is_reported_with_its_pid_and_duration() -> None:
    result = audit(
        5,
        15,
        20,
        [{"pid": 4242, "state": "idle in transaction", "duration_seconds": 42.5, "query": "SELECT 1"}],
    )

    assert result.data.status == "DEGRADED"
    assert len(result.data.hung_queries) == 1
    record = result.data.hung_queries[0]
    assert record.pid == 4242
    assert record.duration_seconds == 42.5
    assert record.state == "idle in transaction"
    assert any("left open" in message for message in result.data.warnings)


def test_a_long_running_active_query_is_reported() -> None:
    result = audit(
        5, 15, 20, [{"pid": 7, "state": "active", "duration_seconds": 30.0, "query": "SELECT pg_sleep(30)"}]
    )

    assert result.data.status == "DEGRADED"
    assert result.data.hung_queries[0].state == "active"


def test_a_short_query_is_not_hung() -> None:
    result = audit(
        5, 15, 20, [{"pid": 7, "state": "active", "duration_seconds": 0.4, "query": "SELECT 1"}]
    )

    assert result.data.hung_queries == []
    assert result.data.status == "HEALTHY"


def test_a_short_idle_transaction_is_not_hung() -> None:
    """Ten seconds is the line; five is a transaction doing its work."""
    result = audit(
        5, 15, 20, [{"pid": 7, "state": "idle in transaction", "duration_seconds": 5.0, "query": "BEGIN"}]
    )

    assert result.data.hung_queries == []


def test_several_stuck_queries_are_critical() -> None:
    result = audit(
        5,
        15,
        20,
        [
            {"pid": 1, "state": "idle in transaction", "duration_seconds": 40.0, "query": "UPDATE t SET a=1"},
            {"pid": 2, "state": "active", "duration_seconds": 30.0, "query": "DELETE FROM t"},
        ],
    )

    assert result.data.status == "CRITICAL"
    assert len(result.data.hung_queries) == 2
    assert result.data.hung_queries[0].duration_seconds >= result.data.hung_queries[1].duration_seconds


def test_empty_or_missing_records_are_safe() -> None:
    for records in (None, [], [None, "not a dict", {}]):
        result = audit(5, 15, 20, records)

        assert result.success is True
        assert result.data.hung_queries == []


# ---------------------------------------------------------------- 3. the scrubbers


@pytest.mark.parametrize(
    "uri",
    [
        "postgresql://app:s3cret@db:5432/indoone",
        "postgres://user:hunter2@host:5432/db",
        "mongodb+srv://u:pa55@cluster/db",
        "redis://default:topsecret@cache:6379/0",
    ],
)
def test_a_password_in_a_uri_is_redacted(uri: str) -> None:
    cleaned = watchdog.sanitize_db_uri(uri)

    assert "[REDACTED]" in cleaned
    for secret in ("s3cret", "hunter2", "pa55", "topsecret"):
        assert secret not in cleaned


def test_a_uri_without_a_password_is_left_alone() -> None:
    assert watchdog.sanitize_db_uri("postgresql://db:5432/indoone") == "postgresql://db:5432/indoone"


@pytest.mark.parametrize(
    ("query", "secret"),
    [
        ("UPDATE users SET password = 'hunter2' WHERE id = 7", "hunter2"),
        ("INSERT INTO otp VALUES ('483920')", "483920"),
        ("SELECT * FROM users WHERE email = 'asha@example.com'", "asha@example.com"),
    ],
)
def test_a_literal_in_a_query_is_redacted(query: str, secret: str) -> None:
    cleaned = watchdog.sanitize_sql_query(query)

    assert secret not in cleaned


def test_the_shape_of_a_query_survives_the_scrub() -> None:
    cleaned = watchdog.sanitize_sql_query("UPDATE users SET password = 'hunter2' WHERE id = 7")

    assert cleaned.startswith("UPDATE users SET password =")
    assert "WHERE id" in cleaned


def test_a_hung_query_preview_is_scrubbed() -> None:
    result = audit(
        5,
        15,
        20,
        [
            {
                "pid": 9,
                "state": "idle in transaction",
                "duration_seconds": 60.0,
                "query": "UPDATE accounts SET balance = 500 WHERE owner = 'asha@example.com'",
            }
        ],
    )

    preview = result.data.hung_queries[0].sanitized_query_preview
    assert "asha@example.com" not in preview
    assert "500" not in preview


# ---------------------------------------------------------------- 4. chips and speed


def test_the_chips_follow_the_status() -> None:
    assert [chip["action"] for chip in audit(5, 15, 20).action_chips] == ["view_pool_metrics"]
    assert [chip["action"] for chip in audit(18, 2, 20).action_chips] == ["inspect_hung_queries"]
    assert [chip["action"] for chip in audit(20, 0, 20).action_chips] == ["recycle_stale_pool"]


def test_the_summary_says_what_is_wrong() -> None:
    result = audit(20, 0, 20)

    assert "CRITICAL" in result.formatted_summary
    assert "Warning" in result.formatted_summary


def test_a_bad_argument_is_refused_cleanly() -> None:
    result = watchdog.audit_pool_state("many", 1, 20)  # type: ignore[arg-type]

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENT"


def test_the_declaration_is_shaped_and_immutable() -> None:
    declarations = watchdog.get_tool_declarations()
    declarations[0]["name"] = "db_drop_everything"

    assert watchdog.get_tool_declarations()[0]["name"] == "db_pool_health"


def test_the_analysis_stays_under_its_budget() -> None:
    records = [
        {"pid": index, "state": "active", "duration_seconds": 1.0, "query": "SELECT 1"}
        for index in range(20)
    ]
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        watchdog.audit_pool_state(5, 15, 20, records)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
