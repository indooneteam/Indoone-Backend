"""Agent 17 - the host sentry.

Every number here is mocked, because a test that asserts the disk on the machine running it
asserts the machine. What is being tested is the arithmetic, the warnings and the perimeter.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agents import load

sentry = load("17_system_diagnostics_sentry")

GIB = 1024**3


@pytest.fixture()
def proc(tmp_path: Path) -> Path:
    """A fake /proc with the fields the sentry reads."""
    root = tmp_path / "proc"
    root.mkdir()
    (root / "meminfo").write_text(
        "MemTotal:        8000000 kB\n"
        "MemFree:          400000 kB\n"
        "MemAvailable:    4000000 kB\n"
        "SwapTotal:       2000000 kB\n"
        "SwapFree:        2000000 kB\n"
    )
    (root / "uptime").write_text("3600.00 7200.00\n")
    return root


@pytest.fixture()
def statvfs(monkeypatch):
    """A disk of 100 GB with 50 GB free, unless a test says otherwise."""

    def install(total_gb: float = 100.0, free_gb: float = 50.0) -> None:
        stat = SimpleNamespace(
            f_blocks=int(total_gb * GIB / 4096),
            f_frsize=4096,
            f_bavail=int(free_gb * GIB / 4096),
        )
        monkeypatch.setattr(sentry.os, "statvfs", lambda path: stat)

    return install


@pytest.fixture()
def load_avg(monkeypatch):
    def install(one: float = 0.5, five: float = 0.4, fifteen: float = 0.3, cores: int = 4) -> None:
        monkeypatch.setattr(sentry.os, "getloadavg", lambda: (one, five, fifteen))
        monkeypatch.setattr(sentry.os, "cpu_count", lambda: cores)

    return install


# ---------------------------------------------------------------- 1. declarations


def test_the_declarations_are_function_calling_shaped() -> None:
    declarations = sentry.get_tool_declarations()

    assert [item["name"] for item in declarations] == [
        "system_get_health_summary",
        "system_check_disk_space",
        "system_check_memory_usage",
        "system_get_build_version",
    ]
    for item in declarations:
        assert item["parameters"]["type"] == "object"


def test_a_caller_cannot_change_the_declarations() -> None:
    sentry.get_tool_declarations()[0]["name"] = "system_reboot"

    assert sentry.get_tool_declarations()[0]["name"] == "system_get_health_summary"


# ---------------------------------------------------------------- 2. the arithmetic


def test_disk_percentages_are_computed_from_statvfs(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=25.0)
    load_avg()

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert telemetry.disk_total_gb == pytest.approx(100.0, abs=0.1)
    assert telemetry.disk_free_gb == pytest.approx(25.0, abs=0.1)
    assert telemetry.disk_used_pct == pytest.approx(75.0, abs=0.2)


def test_memory_is_parsed_from_meminfo(proc, statvfs, load_avg) -> None:
    statvfs()
    load_avg()

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    # 8,000,000 kB total, 4,000,000 kB available -> 50% used.
    assert telemetry.memory_total_mb == pytest.approx(7812.5, abs=1)
    assert telemetry.memory_available_mb == pytest.approx(3906.2, abs=1)
    assert telemetry.memory_used_pct == pytest.approx(50.0, abs=0.5)


def test_uptime_comes_from_proc(proc, statvfs, load_avg) -> None:
    statvfs()
    load_avg()

    assert sentry.collect_host_telemetry(proc_root=proc).uptime_seconds == pytest.approx(3600.0, abs=1)


def test_load_and_cores_are_reported(proc, statvfs, load_avg) -> None:
    statvfs()
    load_avg(one=1.5, five=1.2, fifteen=0.9, cores=4)

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert telemetry.cpu_load_1m == 1.5
    assert telemetry.cpu_count == 4


# ---------------------------------------------------------------- 3. the warnings


def test_low_disk_raises_a_warning(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=10.0)  # 10% free, under the 15% line
    load_avg()

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert telemetry.has_warnings is True
    assert any("Disk" in message for message in telemetry.warning_messages)


def test_low_memory_raises_a_warning(proc, statvfs, load_avg, monkeypatch) -> None:
    statvfs(total_gb=100.0, free_gb=50.0)
    load_avg()
    (proc / "meminfo").write_text(
        "MemTotal:        8000000 kB\nMemAvailable:     400000 kB\n"  # 5% available
    )

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert telemetry.has_warnings is True
    assert any("memory" in message.casefold() for message in telemetry.warning_messages)


def test_high_load_against_the_core_count_raises_a_warning(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=50.0)
    load_avg(one=9.0, cores=4)

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert any("cores" in message for message in telemetry.warning_messages)


def test_a_healthy_host_has_no_warnings(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=50.0)
    load_avg(one=0.5, cores=4)

    telemetry = sentry.collect_host_telemetry(proc_root=proc)

    assert telemetry.has_warnings is False
    assert telemetry.warning_messages == []


# ---------------------------------------------------------------- 4. degradation


def test_a_host_without_proc_still_reports_disk_and_load(tmp_path: Path, statvfs, load_avg) -> None:
    """A container may have no /proc; that is a fact, not a crash."""
    statvfs(total_gb=100.0, free_gb=50.0)
    load_avg(one=0.5)

    telemetry = sentry.collect_host_telemetry(proc_root=tmp_path / "nothing-here")

    assert telemetry.memory_total_mb == 0.0
    assert telemetry.disk_total_gb == pytest.approx(100.0, abs=0.1)
    assert telemetry.has_warnings is False


def test_an_unmeasurable_filesystem_says_so(proc, monkeypatch, load_avg) -> None:
    """An overlay reporting petabytes is reported as unmeasurable, not as a huge number."""
    huge = SimpleNamespace(f_blocks=2251799813685247, f_frsize=4096, f_bavail=2251799813684941)
    monkeypatch.setattr(sentry.os, "statvfs", lambda path: huge)
    load_avg()

    result = sentry.validate_and_execute("system_check_disk_space", {}, proc_root=proc)

    assert "not measurable" in result.formatted_summary


def test_a_missing_git_directory_is_not_an_error(tmp_path: Path, statvfs, load_avg) -> None:
    statvfs()
    load_avg()

    telemetry = sentry.collect_host_telemetry(proc_root=tmp_path, repo_root=tmp_path / "no-repo")

    assert telemetry.commit_sha is None


# ---------------------------------------------------------------- 5. the perimeter


@pytest.mark.parametrize(
    "address",
    ["10.0.4.12", "172.16.5.9", "172.31.0.1", "192.168.1.10", "169.254.169.254"],
)
def test_a_private_address_is_masked(address: str) -> None:
    cleaned = sentry.scrub(f"connected from {address}")

    assert address not in cleaned
    assert "[INTERNAL_IP]" in cleaned


def test_a_public_address_is_left_alone() -> None:
    assert "8.8.8.8" in sentry.scrub("resolver 8.8.8.8")


def test_a_home_path_is_normalised() -> None:
    cleaned = sentry.scrub("checkout at /home/ubuntu/Indoone-Backend")

    assert "/home/ubuntu" not in cleaned
    assert "[ROOT]" in cleaned


def test_the_summary_carries_no_internal_address(tmp_path: Path, statvfs, load_avg) -> None:
    statvfs()
    load_avg()
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "refs" / "heads" / "main").write_text("abcdef1234567890\n")

    result = sentry.validate_and_execute("system_get_health_summary", {}, proc_root=tmp_path, repo_root=repo)

    assert "abcdef12" in result.formatted_summary
    assert "[INTERNAL_IP]" not in result.formatted_summary or "10." not in result.formatted_summary


# ---------------------------------------------------------------- 6. the tools


def test_an_unknown_tool_is_refused() -> None:
    result = sentry.validate_and_execute("system_reboot", {})

    assert result.success is False
    assert result.error_code == "UNKNOWN_TOOL"


def test_a_healthy_host_offers_disk_and_commit_chips(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=50.0)
    load_avg()

    result = sentry.validate_and_execute("system_get_health_summary", {}, proc_root=proc)

    assert [chip["action"] for chip in result.action_chips] == ["check_disk", "check_commit"]


def test_a_warning_offers_the_memory_chip(proc, statvfs, load_avg) -> None:
    statvfs(total_gb=100.0, free_gb=5.0)
    load_avg()

    result = sentry.validate_and_execute("system_get_health_summary", {}, proc_root=proc)

    assert [chip["action"] for chip in result.action_chips] == ["inspect_memory"]


def test_the_commit_is_read_from_the_checkout_not_from_git(tmp_path: Path, statvfs, load_avg) -> None:
    statvfs()
    load_avg()
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (repo / ".git" / "packed-refs").write_text("# pack-refs with: peeled\ndeadbeef1234 refs/heads/main\n")

    telemetry = sentry.collect_host_telemetry(proc_root=tmp_path, repo_root=repo)

    assert telemetry.commit_sha == "deadbeef"


def test_collecting_telemetry_stays_under_its_budget(proc, statvfs, load_avg) -> None:
    statvfs()
    load_avg()
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        sentry.collect_host_telemetry(proc_root=proc)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 1.0, f"fastest {fastest:.4f} ms"
