"""A service request must not be hijacked by the contact parser.

The contact parser's bare find/search/lookup/resolve alternatives matched any
message that began with those verbs - "search my github repositories",
"find my repos", "search my gmail for invoices" - planned a contact lookup, and
the empty contact result then suppressed the model's tool loop. That is why the
assistant answered "I don't have access" instead of calling the GitHub tool.
"""
from __future__ import annotations

import pytest

from app.ai.agent import build_agent_steps

SERVICE_REQUESTS = (
    "search my github repositories",
    "find my repos",
    "search github for the file",
    "find the pull request",
    "search my gmail for invoices",
    "lookup my calendar events",
    "resolve the merge conflict",
)


@pytest.mark.parametrize("message", SERVICE_REQUESTS)
def test_a_service_request_is_not_planned_as_a_contact_lookup(message: str) -> None:
    assert build_agent_steps(message, user_id="u1") == ()


def test_an_explicit_contact_request_still_routes() -> None:
    steps = build_agent_steps("find contact Rahul", user_id="u1")
    assert [step.tool for step in steps] == ["contact_resolve"]


def test_a_call_request_still_routes() -> None:
    steps = build_agent_steps("call Rahul", user_id="u1")
    assert [step.tool for step in steps] == ["phone_call_contact"]
