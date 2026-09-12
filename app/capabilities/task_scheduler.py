from __future__ import annotations

from dataclasses import dataclass


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


def build_schedule(schedule: str, timezone: str = "UTC", retry_limit: int = 3) -> ScheduleDecision:
    normalized = validate_schedule(schedule)
    zone = timezone.strip() or "UTC"
    if len(zone) > 64:
        raise ValueError("timezone exceeds maximum length")
    if retry_limit < 0 or retry_limit > 20:
        raise ValueError("retry_limit must be between 0 and 20")
    return ScheduleDecision(True, normalized, zone, "skip", retry_limit)
