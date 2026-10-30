"""Agent 17 - the host sentry: ask the machine, do not command it.

"Server health check" is a shell command away, and that is exactly the problem. A model
given a shell to answer "how much disk is left" has been given a shell, and the day it writes
``df -h; rm -rf /tmp/*`` the answer to a question nobody asked becomes an outage.

So this asks the operating system directly, and only to read:

* **``os.statvfs``** for disk geometry, **``/proc/meminfo``** for memory, **``os.getloadavg``**
  for load, and **``.git/HEAD``** for the commit that is actually running. No subprocess, no
  shell, no writes - the module imports nothing that can change anything.
* **A missing file is a fact, not a crash.** A container without ``/proc`` still reports
  load and disk; a checkout without ``.git`` still reports health. Every field degrades on
  its own.
* **The perimeter is the point.** An internal address, a username in a path, and the
  environment are what an attacker wants from a health endpoint, so they are masked before
  the text exists rather than filtered afterwards.

The thresholds are stated, not guessed: free disk under fifteen percent, available memory
under ten, and a one-minute load above the core count.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DISK_WARN_FREE_PCT = 15.0
MEMORY_WARN_AVAILABLE_PCT = 10.0
GIB = 1024**3

_PRIVATE_IP = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|169\.254\.\d{1,3}\.\d{1,3})\b"
)
_HOME_PATH = re.compile(r"(?P<prefix>/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)")

_MESSAGES = {
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
    "UNKNOWN_TOOL": {
        "kn": "ಆ ವಿನಂತಿ ನನಗೆ ಗೊತ್ತಿಲ್ಲ.",
        "hi": "वह अनुरोध मुझे ज्ञात नहीं है।",
        "en": "I do not know that request.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "system_get_health_summary",
        "description": "Memory, disk, load and uptime of the host serving this app.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "system_check_disk_space",
        "description": "Free and total disk for a path.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "/"}},
            "required": [],
        },
    },
    {
        "name": "system_check_memory_usage",
        "description": "RAM and swap, from /proc/meminfo.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "system_get_build_version",
        "description": "The commit and branch that are actually running.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
)

_KNOWN_TOOLS = frozenset(item["name"] for item in _TOOL_DECLARATIONS)


@dataclass(slots=True)
class HostTelemetry:
    cpu_load_1m: float = 0.0
    cpu_load_5m: float = 0.0
    cpu_load_15m: float = 0.0
    cpu_count: int = 0
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_used_pct: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_used_pct: float = 0.0
    uptime_seconds: float = 0.0
    commit_sha: str | None = None
    has_warnings: bool = False
    warning_messages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DiagnosticsResult:
    success: bool
    data: HostTelemetry | None = None
    formatted_summary: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_HOST_SENTRY_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_tool_declarations() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- the perimeter


def scrub(text: str) -> str:
    """An internal address and a username are what a health answer must not carry."""
    body = text or ""
    body = _PRIVATE_IP.sub("[INTERNAL_IP]", body)
    body = _HOME_PATH.sub("[ROOT]", body)
    return body


# ---------------------------------------------------------------- harvesting


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _memory(proc_root: Path) -> tuple[float, float]:
    """Total and available RAM in MB, or zeros when there is no /proc to read."""
    body = _read(proc_root / "meminfo")

    if not body:
        return 0.0, 0.0

    values: dict[str, float] = {}

    for line in body.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.strip().split()

        if parts and parts[0].isdigit():
            values[name.strip()] = float(parts[0]) / 1024.0  # kB -> MB

    total = values.get("MemTotal", 0.0)
    available = values.get("MemAvailable", values.get("MemFree", 0.0))

    return total, available


# A container overlay can report a virtual size of petabytes, and printing "8589934592 GB
# free" is worse than saying the figure is not meaningful. Above an exabyte, it is not.
_SANE_DISK_GB = 1e6


def _disk(path: str) -> tuple[float, float]:
    try:
        stat = os.statvfs(path or "/")
    except (OSError, ValueError):
        return 0.0, 0.0

    total = (stat.f_blocks * stat.f_frsize) / GIB
    free = (stat.f_bavail * stat.f_frsize) / GIB

    if total <= 0 or total > _SANE_DISK_GB:
        return 0.0, 0.0

    return total, free


def _load() -> tuple[float, float, float]:
    try:
        one, five, fifteen = os.getloadavg()
        return float(one), float(five), float(fifteen)
    except (OSError, AttributeError):
        return 0.0, 0.0, 0.0


def _uptime(proc_root: Path) -> float:
    """How long the host has been up, from /proc/uptime, or from this process."""
    body = _read(proc_root / "uptime").strip()

    if body:
        first = body.split()[0]
        try:
            return float(first)
        except ValueError:
            pass

    return 0.0


def _commit(repo_root: Path) -> str | None:
    """The running commit, read from the checkout - never by running git."""
    head = _read(repo_root / ".git" / "HEAD").strip()

    if not head:
        return None

    if head.startswith("ref:"):
        ref = head[4:].strip()
        sha = _read(repo_root / ".git" / ref).strip()

        if sha:
            return sha[:8]

        # A packed checkout keeps the ref in packed-refs rather than a file of its own.
        for line in _read(repo_root / ".git" / "packed-refs").splitlines():
            parts = line.split()

            if len(parts) == 2 and parts[1] == ref:
                return parts[0][:8]

        return None

    return head[:8] if re.fullmatch(r"[0-9a-fA-F]{7,40}", head) else None


def collect_host_telemetry(
    *,
    proc_root: str | Path = "/proc",
    repo_root: str | Path | None = None,
    disk_path: str = "/",
) -> HostTelemetry:
    """Everything the sentry can see, with each field degrading on its own."""
    started = time.monotonic()
    proc = Path(proc_root)
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]

    one, five, fifteen = _load()
    total_mb, available_mb = _memory(proc)
    disk_total, disk_free = _disk(disk_path)

    telemetry = HostTelemetry(
        cpu_load_1m=round(one, 2),
        cpu_load_5m=round(five, 2),
        cpu_load_15m=round(fifteen, 2),
        cpu_count=os.cpu_count() or 0,
        memory_total_mb=round(total_mb, 1),
        memory_available_mb=round(available_mb, 1),
        memory_used_pct=round(100.0 * (1 - available_mb / total_mb), 1) if total_mb else 0.0,
        disk_total_gb=round(disk_total, 1),
        disk_free_gb=round(disk_free, 1),
        disk_used_pct=round(100.0 * (1 - disk_free / disk_total), 1) if disk_total else 0.0,
        uptime_seconds=round(_uptime(proc) or (time.monotonic() - started), 1),
        commit_sha=_commit(repo),
    )

    warnings: list[str] = []

    if telemetry.disk_total_gb:
        free_pct = 100.0 * telemetry.disk_free_gb / telemetry.disk_total_gb
        if free_pct < DISK_WARN_FREE_PCT:
            warnings.append(f"Disk is {free_pct:.0f}% free - under the {DISK_WARN_FREE_PCT:.0f}% line.")

    if telemetry.memory_total_mb:
        available_pct = 100.0 * telemetry.memory_available_mb / telemetry.memory_total_mb
        if available_pct < MEMORY_WARN_AVAILABLE_PCT:
            warnings.append(
                f"Only {available_pct:.0f}% of memory is available - under the {MEMORY_WARN_AVAILABLE_PCT:.0f}% line."
            )

    if telemetry.cpu_count and telemetry.cpu_load_1m > telemetry.cpu_count:
        warnings.append(
            f"Load {telemetry.cpu_load_1m} is above the {telemetry.cpu_count} cores available."
        )

    telemetry.warning_messages = warnings
    telemetry.has_warnings = bool(warnings)

    return telemetry


# ---------------------------------------------------------------- the tools


def _summary(telemetry: HostTelemetry) -> str:
    lines = [
        f"Load: {telemetry.cpu_load_1m} / {telemetry.cpu_load_5m} / {telemetry.cpu_load_15m} "
        f"on {telemetry.cpu_count} cores",
        f"Memory: {telemetry.memory_used_pct}% used "
        f"({telemetry.memory_available_mb:.0f} MB free of {telemetry.memory_total_mb:.0f} MB)",
        (
            f"Disk: {telemetry.disk_used_pct}% used "
            f"({telemetry.disk_free_gb:.1f} GB free of {telemetry.disk_total_gb:.1f} GB)"
            if telemetry.disk_total_gb
            else "Disk: not measurable on this filesystem"
        ),
        f"Uptime: {telemetry.uptime_seconds / 3600:.1f} hours",
    ]

    if telemetry.commit_sha:
        lines.append(f"Running commit: {telemetry.commit_sha}")

    if telemetry.warning_messages:
        lines.extend(f"Warning: {message}" for message in telemetry.warning_messages)

    return scrub("\n".join(lines))


def _chips(telemetry: HostTelemetry) -> list[dict[str, str]]:
    if telemetry.has_warnings:
        return [{"label": "Inspect Memory", "action": "inspect_memory"}]

    return [
        {"label": "Check Disk", "action": "check_disk"},
        {"label": "Live Commit", "action": "check_commit"},
    ]


def validate_and_execute(
    call_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    proc_root: str | Path = "/proc",
    repo_root: str | Path | None = None,
    language: str = "en",
) -> DiagnosticsResult:
    """Answer a host question from the operating system, and never change anything."""
    started = time.perf_counter()
    arguments = arguments or {}

    if call_name not in _KNOWN_TOOLS:
        table = _MESSAGES["UNKNOWN_TOOL"]
        key = (language or "en").strip().casefold()[:2]
        return DiagnosticsResult(
            success=False,
            error_code="UNKNOWN_TOOL",
            error_message=table.get(key, table["en"]),
            formatted_summary=table.get(key, table["en"]),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    disk_path = str(arguments.get("path") or "/")

    telemetry = collect_host_telemetry(proc_root=proc_root, repo_root=repo_root, disk_path=disk_path)

    if call_name == "system_check_disk_space":
        summary = scrub(
            (
                f"Disk {disk_path}: {telemetry.disk_free_gb:.1f} GB free of "
                f"{telemetry.disk_total_gb:.1f} GB ({telemetry.disk_used_pct}% used)"
            )
            if telemetry.disk_total_gb
            else f"Disk {disk_path}: not measurable on this filesystem"
        )
    elif call_name == "system_check_memory_usage":
        summary = scrub(
            f"Memory: {telemetry.memory_available_mb:.0f} MB available of "
            f"{telemetry.memory_total_mb:.0f} MB ({telemetry.memory_used_pct}% used)"
        )
    elif call_name == "system_get_build_version":
        summary = scrub(
            f"Running commit: {telemetry.commit_sha}" if telemetry.commit_sha else "The running commit is not readable."
        )
    else:
        summary = _summary(telemetry)

    return DiagnosticsResult(
        success=True,
        data=telemetry,
        formatted_summary=summary,
        action_chips=_chips(telemetry),
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
