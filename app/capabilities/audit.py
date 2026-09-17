from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    user_id: str
    actor: str
    resource: str
    action: str
    outcome: str
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @classmethod
    def create(
        cls,
        event_type: str,
        user_id: str,
        actor: str,
        resource: str,
        action: str,
        outcome: str,
        request_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "AuditEvent":
        return cls(
            event_id=str(uuid4()),
            event_type=event_type,
            user_id=user_id.strip(),
            actor=actor.strip(),
            resource=resource.strip(),
            action=action.strip(),
            outcome=outcome.strip(),
            request_id=request_id.strip(),
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def as_record(self) -> dict[str, Any]:
        return asdict(self)
