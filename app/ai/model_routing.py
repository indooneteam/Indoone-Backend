from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    task: str
    model_role: str
    reason: str


def route_task(task: str) -> ModelRoute:
    normalized = task.strip().lower()
    if any(word in normalized for word in ("image", "photo", "vision", "screenshot")):
        return ModelRoute(normalized, "vision", "visual input detected")
    if any(word in normalized for word in ("code", "python", "javascript", "debug", "repository")):
        return ModelRoute(normalized, "coding", "software task detected")
    if any(word in normalized for word in ("search", "research", "latest", "news", "current")):
        return ModelRoute(normalized, "research", "fresh-information task detected")
    if any(word in normalized for word in ("voice", "audio", "speak", "listen")):
        return ModelRoute(normalized, "speech", "speech task detected")
    return ModelRoute(normalized, "general", "general conversational task")
