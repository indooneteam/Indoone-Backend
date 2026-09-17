from __future__ import annotations

from dataclasses import dataclass

from app.capabilities.contacts import normalize_phone


@dataclass(frozen=True)
class CallAction:
    phone_number: str
    contact_name: str = ""
    requires_confirmation: bool = True

    def as_dict(self) -> dict[str, object]:
        label = self.contact_name or self.phone_number
        return {
            "type": "phone.call",
            "phone_number": self.phone_number,
            "contact_name": self.contact_name,
            "requires_confirmation": self.requires_confirmation,
            "confirmation_message": f"Call {label}?",
        }


def build_call_action(phone_number: object, contact_name: object = "") -> CallAction:
    normalized = normalize_phone(phone_number)
    name = " ".join(str(contact_name or "").split()).strip()
    if len(name) > 200:
        raise ValueError("contact_name is too long")
    return CallAction(phone_number=normalized, contact_name=name)
