from __future__ import annotations

import pytest

from app.capabilities.contacts import parse_contacts, resolve_contact, search_contacts
from app.capabilities.phone import build_call_action


def _contacts() -> list[dict[str, str]]:
    return [
        {"contact_id": "1", "name": "Rahul", "phone": "+91 98765 43210", "email": "rahul@example.com"},
        {"contact_id": "2", "name": "Ravi Kumar", "phone": "080-1234-5678"},
        {"contact_id": "3", "name": "Anita", "phone": "+91 99887 77665"},
    ]


def test_phone_normalization_and_contact_search() -> None:
    contacts = parse_contacts(_contacts())
    matches = search_contacts(contacts, "rah")
    assert [item.name for item in matches] == ["Rahul"]
    assert contacts[0].phone == "+919876543210"


def test_contact_resolve_prefers_exact_name() -> None:
    contacts = parse_contacts(_contacts())
    result = resolve_contact(contacts, "Rahul")
    assert result is not None
    assert result.phone == "+919876543210"


def test_call_action_requires_confirmation() -> None:
    action = build_call_action("+91 98765 43210", "Rahul")
    payload = action.as_dict()
    assert payload["type"] == "phone.call"
    assert payload["phone_number"] == "+919876543210"
    assert payload["requires_confirmation"] is True


def test_invalid_phone_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_call_action("123")
