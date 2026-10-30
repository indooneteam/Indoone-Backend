"""Agent 20 - the Instagram DM adapter.

Three things are being protected here: the signature that proves a DM came from Meta, the
echo that would otherwise make the bot answer itself forever, and the thousand-character
boundary that must not cut a Kannada letter in half.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from app.agents import load

instagram = load("20_instagram_dm_adapter")

SECRET = "instagram-app-secret"


@pytest.fixture(autouse=True)
def _fresh():
    instagram.reset_dedup_cache()
    yield
    instagram.reset_dedup_cache()


def sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def dm(mid: str = "m.1", **message) -> str:
    return json.dumps(
        {
            "entry": [
                {
                    "messaging": [
                        {
                            "sender": {"id": "17841400123456789"},
                            "timestamp": 1720000000000,
                            "message": {"mid": mid, "text": "naale raja na?", **message},
                        }
                    ]
                }
            ]
        }
    )


# ---------------------------------------------------------------- 1. the door


def test_a_signed_body_verifies() -> None:
    body = b'{"entry":[]}'

    assert instagram.verify_signature(body, sign(body), SECRET) is True


def test_an_unsigned_wrongly_signed_or_altered_body_fails() -> None:
    body = b'{"entry":[]}'

    assert instagram.verify_signature(body, None, SECRET) is False
    assert instagram.verify_signature(body, "sha256=" + "0" * 64, SECRET) is False
    assert instagram.verify_signature(body, sign(body), "another-secret") is False
    assert instagram.verify_signature(body + b" ", sign(body), SECRET) is False


def test_a_missing_secret_fails_closed() -> None:
    body = b'{"entry":[]}'

    assert instagram.verify_signature(body, sign(body), "") is False


# ---------------------------------------------------------------- 2. the flattening


def test_a_dm_yields_mid_identity_text_and_timestamp() -> None:
    result = instagram.parse_and_normalize(dm())

    assert result.is_valid is True
    assert result.status_code == 200
    message = result.normalized_message
    assert message.mid == "m.1"
    assert message.channel_user_id == "ig:17841400123456789"
    assert message.message_type == "text"
    assert message.text_content == "naale raja na?"
    assert message.timestamp_epoch == 1720000000


def test_a_bot_echo_is_silenced() -> None:
    """Processing an echo means answering yourself, then answering that answer."""
    payload = dm("m.2", is_echo=True, text="here is your answer")

    message = instagram.parse_and_normalize(payload).normalized_message

    assert message.message_type == "echo"
    assert message.is_echo is True


def test_a_quick_reply_carries_its_payload() -> None:
    payload = dm("m.3", text="Confirm", quick_reply={"payload": "confirm_send:abc123"})

    message = instagram.parse_and_normalize(payload).normalized_message

    assert message.message_type == "quick_reply"
    assert message.action_payload == "confirm_send:abc123"


def test_an_attachment_carries_a_pointer_and_no_blob() -> None:
    payload = dm("m.4", attachments=[{"type": "story_mention", "url": "https://cdn.example/x.jpg"}])

    message = instagram.parse_and_normalize(payload).normalized_message

    assert message.message_type == "attachment"
    assert message.attachment_type == "story_mention"
    assert message.attachment_url == "https://cdn.example/x.jpg"


@pytest.mark.parametrize("key", ["read", "delivery", "reaction"])
def test_a_system_event_is_acknowledged_not_refused(key: str) -> None:
    payload = json.dumps({"entry": [{"messaging": [{"sender": {"id": "1"}, key: {"mid": "x"}}]}]})

    result = instagram.parse_and_normalize(payload)

    assert result.is_valid is True
    assert result.status_code == 200
    assert result.normalized_message.message_type == "system"


# ---------------------------------------------------------------- 3. the window


def test_a_retry_of_the_same_mid_is_a_duplicate() -> None:
    payload = dm("m.5")

    first = instagram.parse_and_normalize(payload)
    second = instagram.parse_and_normalize(payload)

    assert first.normalized_message.is_duplicate is False
    assert second.normalized_message.is_duplicate is True


def test_a_retry_after_the_window_is_new() -> None:
    payload = dm("m.6")
    now = time.time()

    instagram.parse_and_normalize(payload, now=now)
    later = instagram.parse_and_normalize(payload, now=now + instagram.DEDUP_TTL_SECONDS + 1)

    assert later.normalized_message.is_duplicate is False


def test_the_window_stays_bounded() -> None:
    now = time.time()

    for index in range(instagram.DEDUP_MAXSIZE + 200):
        instagram.parse_and_normalize(dm(f"m.{index}"), now=now)

    assert len(instagram._seen) <= instagram.DEDUP_MAXSIZE


# ---------------------------------------------------------------- 4. egress


def test_markdown_is_stripped() -> None:
    rendered = instagram.format_for_instagram(
        "## Heading\n**bold** and *italic* and [link](https://example.com)\n---\n> quoted"
    )

    assert "##" not in rendered
    assert "**" not in rendered
    assert rendered.count("*") == 0
    assert "bold" in rendered and "italic" in rendered
    assert "link (https://example.com)" in rendered
    assert ">" not in rendered
    assert "---" not in rendered


def test_an_ordinary_message_is_left_readable() -> None:
    assert instagram.format_for_instagram("Deepavali is on 8 November.") == "Deepavali is on 8 November."


def test_a_long_message_is_clamped_inside_the_boundary() -> None:
    rendered = instagram.format_for_instagram("word " * 400)

    assert len(rendered) <= instagram.INSTAGRAM_LIMIT


@pytest.mark.parametrize(
    "text",
    [
        "ಕನ್ನಡ ರಾಜ್ಯೋತ್ಸವ " * 100,
        "ರಾಷ್ಟ್ರೀಯ ದಿನಾಂಕ " * 100,
        "हिन्दी भाषा का महत्व " * 100,
        "தமிழ் மொழி வரலாறு " * 100,
    ],
)
def test_a_clamped_message_never_ends_on_a_broken_conjunct(text: str) -> None:
    """The last character must not be a combining mark or an orphaned virama."""
    import unicodedata

    rendered = instagram.format_for_instagram(text)
    body = rendered[:-1] if rendered.endswith("…") else rendered

    assert len(rendered) <= instagram.INSTAGRAM_LIMIT
    assert not unicodedata.combining(body[-1]) if body else True
    assert body[-1] not in {"\u0ccd", "\u094d", "\u0c4d", "\u0bcd"}


# ---------------------------------------------------------------- 5. bad input and speed


@pytest.mark.parametrize("payload", ["", "   ", "{not json", "[]", "null"])
def test_unreadable_input_is_a_clean_400(payload: str) -> None:
    result = instagram.parse_and_normalize(payload)

    assert result.is_valid is False
    assert result.status_code == 400
    assert result.error_message


def test_an_entry_with_no_message_is_refused() -> None:
    payload = json.dumps({"entry": [{"messaging": [{"sender": {"id": "1"}}]}]})

    assert instagram.parse_and_normalize(payload).status_code == 400


def test_parsing_stays_under_its_budget() -> None:
    payload = dm("m.99")
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        instagram.parse_and_normalize(payload, check_duplicate=False)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_formatting_stays_under_its_budget() -> None:
    text = "**bold** and [link](https://x.com) " * 40
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        instagram.format_for_instagram(text)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
