"""The contact tools must never be offered to the model.

They take a client-supplied contact snapshot. The model has none, so offering the
tool only invited it to invent contacts - and it always returned an empty result.
The tools stay in the registry for the deterministic agent path and the client
API, which do supply contacts.
"""
from __future__ import annotations

import json

from app.ai.tool_loop import model_tool_declarations
from app.ai.tool_registry import get_tool_spec
from app.ai.tools import run_tool

_CONTACT_TOOLS = ("contacts_search", "contact_resolve", "phone_call_contact")


def test_the_contact_tools_are_not_offered_to_the_model():
    names = {item["name"] for item in model_tool_declarations()}
    for name in _CONTACT_TOOLS:
        assert name not in names, name
    # phone_call takes a number from the message, not a contact snapshot, so it stays.
    assert "phone_call" in names


def test_the_contact_tools_remain_for_the_agent_path():
    for name in _CONTACT_TOOLS:
        spec = get_tool_spec(name)
        assert spec is not None, name
        assert spec.model_facing is False, name


def test_a_contact_tool_still_resolves_a_client_supplied_snapshot():
    payload = json.dumps({
        "contacts": [{"name": "Rahul", "phone": "+919000000000"}],
        "query": "Rahul",
    })
    result = run_tool("contact_resolve", payload)
    assert result.safe is True
    assert json.loads(result.output)["contact"]["name"] == "Rahul"
