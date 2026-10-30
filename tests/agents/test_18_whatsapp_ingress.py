"""Agent 18 - the WhatsApp ingress.

Two properties. The first is a security boundary: a body that Meta did not sign must never
become a message from a phone number. The second is a correctness one: Meta re-sends
webhooks, and a retry must not become a second answer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from app.agents import load

ingress = load("18_whatsapp_ingress")

SECRET = "app-secret-value"


@pytest.fixture(autouse=True)
def _fresh():
    ingress.reset_dedup_cache()
    yield
    ingress.reset_dedup_cache()


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def webhook(message: dict, *, contact: dict | None = None) -> str:
    return json.dumps(
        {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [message],
                                "contacts": [contact or {"profile": {"name": "Asha"}, "wa_id": "919876543210"}],
                            }
                        }
                    ]
                }
            ]
        }
    )


def text_message(wamid: str = "wamid.1", body: str = "naale raja na?") -> dict:
    return {
        "id": wamid,
        "from": "+91 98765-43210",
        "timestamp": "1720000000",
        "type": "text",
        "text": {"body": body},
    }


# ---------------------------------------------------------------- 1. the signature


def test_a_signed_body_verifies() -> None:
    body = b'{"entry":[]}'

    assert ingress.verify_webhook_signature(body, sign(body), SECRET) is True


def test_an_unsigned_or_wrongly_signed_body_fails() -> None:
    body = b'{"entry":[]}'

    assert ingress.verify_webhook_signature(body, None, SECRET) is False
    assert ingress.verify_webhook_signature(body, "sha256=" + "0" * 64, SECRET) is False
    assert ingress.verify_webhook_signature(body, "sha1=abc", SECRET) is False
    assert ingress.verify_webhook_signature(body, sign(body), "a-different-secret") is False


def test_a_missing_secret_fails_closed() -> None:
    body = b'{"entry":[]}'

    assert ingress.verify_webhook_signature(body, sign(body), "") is False


def test_a_body_changed_after_signing_fails() -> None:
    body = b'{"entry":[{"changes":[]}]}'
    signature = sign(body)

    assert ingress.verify_webhook_signature(body + b" ", signature, SECRET) is False


# ---------------------------------------------------------------- 2. the flattening


def test_a_text_message_yields_a_canonical_identity_and_clean_text() -> None:
    result = ingress.parse_and_normalize(webhook(text_message()))

    assert result.is_valid is True
    assert result.status_code == 200
    message = result.normalized_message
    assert message is not None
    assert message.wamid == "wamid.1"
    assert message.channel_user_id == "wa:919876543210"
    assert message.sender_name == "Asha"
    assert message.message_type == "text"
    assert message.text_content == "naale raja na?"
    assert message.timestamp_epoch == 1720000000
    assert message.is_duplicate is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+91 98765-43210", "wa:919876543210"),
        ("919876543210", "wa:919876543210"),
        ("9876543210", "wa:919876543210"),
        ("+91-98765 43210", "wa:919876543210"),
        ("", ""),
    ],
)
def test_a_phone_number_becomes_one_canonical_id(raw: str, expected: str) -> None:
    assert ingress.canonical_channel_id(raw) == expected


def test_a_button_reply_carries_its_action_token() -> None:
    payload = webhook(
        {
            "id": "wamid.2",
            "from": "919876543210",
            "type": "interactive",
            "interactive": {"button_reply": {"id": "confirm_send:abc123", "title": "Confirm & Send"}},
        }
    )

    message = ingress.parse_and_normalize(payload).normalized_message

    assert message is not None
    assert message.message_type == "interactive_button"
    assert message.action_id == "confirm_send:abc123"
    assert message.text_content == "Confirm & Send"


def test_a_list_reply_carries_the_chosen_item() -> None:
    payload = webhook(
        {
            "id": "wamid.3",
            "from": "919876543210",
            "type": "interactive",
            "interactive": {"list_reply": {"id": "month_schedule", "title": "Month View"}},
        }
    )

    message = ingress.parse_and_normalize(payload).normalized_message

    assert message is not None
    assert message.message_type == "interactive_list"
    assert message.action_id == "month_schedule"


def test_a_media_message_carries_a_pointer_and_never_a_blob() -> None:
    payload = webhook(
        {
            "id": "wamid.4",
            "from": "919876543210",
            "type": "image",
            "image": {"id": "media-123", "mime_type": "image/jpeg", "caption": "the receipt"},
        }
    )

    message = ingress.parse_and_normalize(payload).normalized_message

    assert message is not None
    assert message.message_type == "media"
    assert message.media_id == "media-123"
    assert message.media_mime_type == "image/jpeg"
    assert message.text_content == "the receipt"


def test_a_delivery_receipt_is_acknowledged_without_being_a_message() -> None:
    payload = json.dumps({"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]})

    result = ingress.parse_and_normalize(payload)

    assert result.is_valid is True
    assert result.status_code == 200
    assert result.normalized_message is not None
    assert result.normalized_message.message_type == "status"


# ---------------------------------------------------------------- 3. the window


def test_a_retry_of_the_same_wamid_is_a_duplicate() -> None:
    payload = webhook(text_message("wamid.5"))

    first = ingress.parse_and_normalize(payload)
    second = ingress.parse_and_normalize(payload)

    assert first.normalized_message.is_duplicate is False
    assert second.normalized_message.is_duplicate is True


def test_a_different_wamid_is_not_a_duplicate() -> None:
    ingress.parse_and_normalize(webhook(text_message("wamid.6")))
    second = ingress.parse_and_normalize(webhook(text_message("wamid.7")))

    assert second.normalized_message.is_duplicate is False


def test_a_retry_after_the_window_is_a_new_message() -> None:
    payload = webhook(text_message("wamid.8"))
    now = time.time()

    ingress.parse_and_normalize(payload, now=now)
    later = ingress.parse_and_normalize(payload, now=now + ingress.DEDUP_TTL_SECONDS + 1)

    assert later.normalized_message.is_duplicate is False


def test_the_window_is_bounded() -> None:
    """A set that grows forever is a leak; 1024 ids is the ceiling."""
    now = time.time()

    for index in range(ingress.DEDUP_MAXSIZE + 200):
        ingress.parse_and_normalize(webhook(text_message(f"wamid.{index}")), now=now)

    assert len(ingress._seen) <= ingress.DEDUP_MAXSIZE


def test_duplicate_checking_can_be_skipped() -> None:
    payload = webhook(text_message("wamid.9"))

    ingress.parse_and_normalize(payload)
    again = ingress.parse_and_normalize(payload, check_duplicate=False)

    assert again.normalized_message.is_duplicate is False


# ---------------------------------------------------------------- 4. bad input


@pytest.mark.parametrize("payload", ["", "   ", "{not json", "[]", "null"])
def test_unreadable_input_is_a_clean_400(payload: str) -> None:
    result = ingress.parse_and_normalize(payload)

    assert result.is_valid is False
    assert result.status_code == 400
    assert result.error_message
    assert result.normalized_message is None


def test_a_payload_with_no_message_is_refused() -> None:
    result = ingress.parse_and_normalize(json.dumps({"entry": [{"changes": [{"value": {}}]}]}))

    assert result.is_valid is False
    assert result.status_code == 400


# ---------------------------------------------------------------- 5. speed


def test_parsing_stays_under_its_budget() -> None:
    payload = webhook(text_message("wamid.speed"))
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        ingress.parse_and_normalize(payload, check_duplicate=False)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
