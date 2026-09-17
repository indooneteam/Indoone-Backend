from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

MAX_CONTACTS = 500
MAX_NAME_LENGTH = 200
MAX_PHONE_LENGTH = 32
MAX_EMAIL_LENGTH = 320

_PHONE_CLEAN_RE = re.compile(r"[^0-9+]")


@dataclass(frozen=True)
class Contact:
    name: str
    phone: str
    email: str = ""
    contact_id: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "contact_id": self.contact_id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
        }


def normalize_phone(value: object) -> str:
    phone = str(value or "").strip()
    if not phone:
        raise ValueError("phone cannot be empty")
    phone = _PHONE_CLEAN_RE.sub("", phone)
    if phone.startswith("00"):
        phone = "+" + phone[2:]
    if phone.startswith("+"):
        digits = phone[1:]
        if not digits.isdigit():
            raise ValueError("phone must contain digits after the country code")
        normalized = "+" + digits
    else:
        if not phone.isdigit():
            raise ValueError("phone must contain only digits or a leading +")
        normalized = phone
    if len(normalized) > MAX_PHONE_LENGTH or len(normalized.lstrip("+")) < 7:
        raise ValueError("phone number length is invalid")
    return normalized


def _text(value: object, field: str, max_length: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    if len(text) > max_length:
        raise ValueError(f"{field} is too long")
    return text


def parse_contact(payload: Any) -> Contact:
    if not isinstance(payload, dict):
        raise ValueError("contact must be an object")
    name = _text(payload.get("name"), "name", MAX_NAME_LENGTH)
    phone = normalize_phone(payload.get("phone"))
    email = str(payload.get("email") or "").strip()
    if len(email) > MAX_EMAIL_LENGTH:
        raise ValueError("email is too long")
    contact_id = str(payload.get("contact_id") or "").strip()
    if len(contact_id) > 128:
        raise ValueError("contact_id is too long")
    return Contact(name=name, phone=phone, email=email, contact_id=contact_id)


def parse_contacts(payloads: list[dict[str, Any]]) -> list[Contact]:
    if len(payloads) > MAX_CONTACTS:
        raise ValueError(f"too many contacts; maximum is {MAX_CONTACTS}")
    return [parse_contact(item) for item in payloads]


def search_contacts(contacts: list[Contact], query: str, limit: int = 20) -> list[Contact]:
    query = " ".join(query.strip().split()).casefold()
    if not query:
        return contacts[:limit]
    normalized_phone_query = re.sub(r"\D", "", query)
    scored: list[tuple[int, Contact]] = []
    for contact in contacts:
        name = contact.name.casefold()
        email = contact.email.casefold()
        phone_digits = re.sub(r"\D", "", contact.phone)
        score = 0
        if name == query:
            score = 100
        elif name.startswith(query):
            score = 80
        elif query in name:
            score = 60
        elif normalized_phone_query and normalized_phone_query in phone_digits:
            score = 50
        elif query in email:
            score = 40
        if score:
            scored.append((score, contact))
    scored.sort(key=lambda item: (-item[0], item[1].name.casefold()))
    return [contact for _, contact in scored[:limit]]


def resolve_contact(contacts: list[Contact], query: str) -> Contact | None:
    matches = search_contacts(contacts, query, limit=2)
    return matches[0] if matches else None
