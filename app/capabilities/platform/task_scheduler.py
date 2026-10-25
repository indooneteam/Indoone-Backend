from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_MISFIRE_POLICIES = frozenset({"skip", "run_once"})


@dataclass(frozen=True)
class ScheduleDecision:
    enabled: bool
    schedule: str
    timezone: str = "UTC"
    misfire_policy: str = "skip"
    retry_limit: int = 3


def validate_schedule(schedule: str) -> str:
    value = schedule.strip()
    if not value:
        raise ValueError("schedule cannot be empty")
    if len(value) > 512:
        raise ValueError("schedule exceeds maximum length")
    return value


def build_schedule(
    schedule: str,
    timezone: str = "UTC",
    retry_limit: int = 3,
    misfire_policy: str = "skip",
) -> ScheduleDecision:
    normalized = validate_schedule(schedule)
    zone = timezone.strip() or "UTC"
    if len(zone) > 64:
        raise ValueError("timezone exceeds maximum length")
    if retry_limit < 0 or retry_limit > 20:
        raise ValueError("retry_limit must be between 0 and 20")
    policy = misfire_policy.strip().lower() or "skip"
    if policy not in _ALLOWED_MISFIRE_POLICIES:
        raise ValueError("misfire_policy must be skip or run_once")
    return ScheduleDecision(True, normalized, zone, policy, retry_limit)
