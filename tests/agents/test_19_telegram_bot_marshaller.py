"""Agent 19 - the Telegram ingress.

Same two properties as the WhatsApp door - a secret that proves origin, a window that kills
the retry - plus one that is Telegram's own: a slash command is not a sentence.
"""

from __future__ import annotations

import json
import time

import pytest

from app.agents import load

telegram = load("19_telegram_bot_marshaller")

SECRET = "webhook-secret-value"


@pytest.fixture(autouse=True)
def _fresh():
    telegram.reset_dedup_cache()
    yield
    telegram.reset_dedup_cache()


def sender(user_id: int = 987654321) -> dict:
    return {"id": user_id, "first_name": "Asha", "last_name": "R", "username": "asha"}


def update(update_id: int = 1, **payload) -> str:
    return json.dumps({"update_id": update_id, **payload})


def message(update_id: int = 1, **fields) -> str:
    body = {"date": 1720000000, "from": sender(), **fields}
    return update(update_id, message=body)


# ---------------------------------------------------------------- 1. the door


def test_a_matching_secret_passes() -> None:
    assert telegram.verify_secret_token(SECRET, SECRET) is True


def test_a_mismatched_missing_or_empty_secret_fails() -> None:
    assert telegram.verify_secret_token("wrong", SECRET) is False
    assert telegram.verify_secret_token(None, SECRET) is False
    assert telegram.verify_secret_token("", SECRET) is False
    assert telegram.verify_secret_token(SECRET, "") is False


def test_a_secret_differing_in_one_character_fails() -> None:
    assert telegram.verify_secret_token(SECRET[:-1] + "X", SECRET) is False


# ---------------------------------------------------------------- 2. the flattening


def test_a_text_message_yields_identity_and_text() -> None:
    result = telegram.parse_and_normalize(message(text="naale raja na?"))

    assert result.is_valid is True
    assert result.status_code == 200
    normalized = result.normalized_message
    assert normalized.update_id == 1
    assert normalized.channel_user_id == "tg:987654321"
    assert normalized.sender_name == "Asha R"
    assert normalized.username == "asha"
    assert normalized.message_type == "text"
    assert normalized.text_content == "naale raja na?"
    assert normalized.timestamp_epoch == 1720000000
    assert normalized.is_duplicate is False


@pytest.mark.parametrize(
    ("text", "command", "args"),
    [
        ("/start arg1", "start", "arg1"),
        ("/help", "help", ""),
        ("/reset@IndooneBot now", "reset", "now"),
        ("/status   two  spaces", "status", "two  spaces"),
    ],
)
def test_a_bot_command_decomposes_into_a_name_and_its_arguments(
    text: str, command: str, args: str
) -> None:
    normalized = telegram.parse_and_normalize(message(text=text)).normalized_message

    assert normalized.message_type == "command"
    assert normalized.command == command
    assert normalized.command_args == args


def test_a_slash_in_the_middle_of_a_sentence_is_not_a_command() -> None:
    normalized = telegram.parse_and_normalize(message(text="what is 10/2?")).normalized_message

    assert normalized.message_type == "text"
    assert normalized.command is None


def test_an_inline_button_press_keeps_the_callback_id_and_the_action() -> None:
    payload = update(
        2,
        callback_query={"id": "cb-1", "from": sender(42), "data": "confirm_send:abc123"},
    )

    normalized = telegram.parse_and_normalize(payload).normalized_message

    assert normalized.message_type == "callback_query"
    assert normalized.callback_query_id == "cb-1"
    assert normalized.text_content == "confirm_send:abc123"
    assert normalized.channel_user_id == "tg:42"


def test_a_voice_note_carries_a_pointer_and_no_blob() -> None:
    payload = message(3, voice={"file_id": "voice-1", "mime_type": "audio/ogg"})

    normalized = telegram.parse_and_normalize(payload).normalized_message

    assert normalized.message_type == "media"
    assert normalized.media_file_id == "voice-1"
    assert normalized.media_mime_type == "audio/ogg"


def test_a_photo_takes_the_largest_size() -> None:
    payload = message(
        4,
        photo=[{"file_id": "small"}, {"file_id": "medium"}, {"file_id": "large"}],
        caption="the receipt",
    )

    normalized = telegram.parse_and_normalize(payload).normalized_message

    assert normalized.media_file_id == "large"
    assert normalized.text_content == "the receipt"


def test_a_document_carries_its_mime_type() -> None:
    payload = message(5, document={"file_id": "doc-1", "mime_type": "application/pdf"})

    normalized = telegram.parse_and_normalize(payload).normalized_message

    assert normalized.media_file_id == "doc-1"
    assert normalized.media_mime_type == "application/pdf"


# ---------------------------------------------------------------- 3. the window


def test_a_retry_of_the_same_update_id_is_a_duplicate() -> None:
    payload = message(10, text="hello")

    first = telegram.parse_and_normalize(payload)
    second = telegram.parse_and_normalize(payload)

    assert first.normalized_message.is_duplicate is False
    assert second.normalized_message.is_duplicate is True


def test_a_different_update_id_is_not_a_duplicate() -> None:
    telegram.parse_and_normalize(message(11, text="one"))
    second = telegram.parse_and_normalize(message(12, text="two"))

    assert second.normalized_message.is_duplicate is False


def test_a_retry_after_the_window_is_new() -> None:
    payload = message(13, text="hello")
    now = time.time()

    telegram.parse_and_normalize(payload, now=now)
    later = telegram.parse_and_normalize(payload, now=now + telegram.DEDUP_TTL_SECONDS + 1)

    assert later.normalized_message.is_duplicate is False


def test_the_window_stays_bounded() -> None:
    now = time.time()

    for index in range(telegram.DEDUP_MAXSIZE + 200):
        telegram.parse_and_normalize(message(index, text="x"), now=now)

    assert len(telegram._seen) <= telegram.DEDUP_MAXSIZE


# ---------------------------------------------------------------- 4. everything else


@pytest.mark.parametrize("key", ["channel_post", "edited_message", "my_chat_member", "edited_channel_post"])
def test_a_non_user_update_is_acknowledged_not_refused(key: str) -> None:
    result = telegram.parse_and_normalize(update(20, **{key: {"date": 1}}))

    assert result.is_valid is True
    assert result.status_code == 200
    assert result.normalized_message.message_type == "unsupported"


@pytest.mark.parametrize("payload", ["", "   ", "{not json", "[]", "null"])
def test_unreadable_input_is_a_clean_400(payload: str) -> None:
    result = telegram.parse_and_normalize(payload)

    assert result.is_valid is False
    assert result.status_code == 400
    assert result.error_message
    assert result.normalized_message is None


def test_an_update_with_nothing_in_it_is_refused() -> None:
    result = telegram.parse_and_normalize(update(30))

    assert result.is_valid is False
    assert result.status_code == 400


# ---------------------------------------------------------------- 5. speed


def test_parsing_stays_under_its_budget() -> None:
    payload = message(40, text="naale raja na?")
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        telegram.parse_and_normalize(payload, check_duplicate=False)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_the_channel_client_also_compares_its_secret_in_constant_time() -> None:
    """The existing client compared with ==, which leaks how much of the secret was right."""
    import inspect

    from app.channels.telegram import service

    source = inspect.getsource(service.validate_webhook_secret)

    assert "compare_digest" in source
