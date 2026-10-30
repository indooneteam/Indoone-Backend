"""Agent 12 - the mail supervisor.

The property that matters is one sentence: **the model may ask to send, and may not send.**
Everything else here supports it - the token, the session binding, the single use - or
protects the other direction, an OTP arriving in the inbox and leaving in a prompt.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from app.agents import load

mail = load("12_zoho_mail_supervisor")


@pytest.fixture(autouse=True)
def _fresh():
    mail.reset()
    yield
    mail.reset()


def run(call, session="s1", **kwargs):
    return asyncio.run(mail.validate_and_execute(call, session, **kwargs))


async def _fake_client(tool, arguments, token):
    if tool == "zoho_draft_email":
        return {"draft_id": "draft-1"}
    if tool == "zoho_send_email":
        return {"sent": True}
    return {"ok": True}


def _draft(**overrides):
    arguments = {"to": ["asha@example.com"], "subject": "Hello", "body": "Text"}
    arguments.update(overrides)
    return run(mail.MailToolCall("zoho_draft_email", arguments), executor=_fake_client)


# ---------------------------------------------------------------- 1. declarations


def test_the_declarations_are_function_calling_shaped() -> None:
    declarations = mail.get_tool_declarations()

    assert [item["name"] for item in declarations] == [
        "zoho_search_emails",
        "zoho_read_thread",
        "zoho_draft_email",
        "zoho_send_email",
    ]
    for item in declarations:
        assert item["parameters"]["type"] == "object"
        assert item["parameters"]["required"]


def test_a_caller_cannot_change_the_declarations() -> None:
    mail.get_tool_declarations()[0]["name"] = "zoho_delete_everything"

    assert mail.get_tool_declarations()[0]["name"] == "zoho_search_emails"


# ---------------------------------------------------------------- 2. arguments


@pytest.mark.parametrize(
    "address",
    ["not-an-email", "a@b", "@example.com", "a b@example.com", "a@@example.com"],
)
def test_a_malformed_address_is_refused(address: str) -> None:
    arguments, problem = mail.validate_arguments(
        mail.MailToolCall("zoho_draft_email", {"to": [address], "subject": "s", "body": "b"})
    )

    assert arguments is None
    assert "invalid email address" in problem


def test_an_empty_recipient_list_is_refused() -> None:
    arguments, problem = mail.validate_arguments(
        mail.MailToolCall("zoho_draft_email", {"to": [], "subject": "s", "body": "b"})
    )

    assert arguments is None
    assert "no recipients" in problem


def test_an_empty_subject_is_refused() -> None:
    arguments, problem = mail.validate_arguments(
        mail.MailToolCall("zoho_draft_email", {"to": ["a@b.com"], "subject": "  ", "body": "b"})
    )

    assert arguments is None
    assert "missing argument" in problem


def test_a_search_limit_is_clamped() -> None:
    arguments, _ = mail.validate_arguments(
        mail.MailToolCall("zoho_search_emails", {"query": "invoice", "limit": "999"})
    )

    assert arguments["limit"] == mail.MAX_SEARCH_RESULTS


# ---------------------------------------------------------------- 3. the gate


def test_a_draft_asks_for_confirmation_and_offers_chips() -> None:
    result = _draft()

    assert result.success is True
    assert result.requires_confirmation is True
    assert result.confirmation_token
    actions = [chip["action"] for chip in result.action_chips]
    assert f"confirm_send:{result.confirmation_token}" in actions
    assert "discard_draft" in actions


def test_the_model_never_sees_the_token() -> None:
    """A model that can read the token can send without anyone agreeing."""
    result = _draft()

    assert result.confirmation_token not in result.sanitized_output
    assert "Nothing has been sent" in result.sanitized_output


def test_sending_without_a_token_is_blocked() -> None:
    result = run(
        mail.MailToolCall("zoho_send_email", {"draft_id": "draft-1", "confirmation_token": "deadbe"}),
        executor=_fake_client,
    )

    assert result.success is False
    assert result.error_code == "UNAUTHORIZED_MUTATION"


def test_sending_with_the_matching_token_succeeds() -> None:
    draft = _draft()

    result = mail.verify_and_dispatch(
        "draft-1", draft.confirmation_token, "s1", executor=lambda *a: {"sent": True}
    )

    assert result.success is True


def test_a_token_is_bound_to_its_draft() -> None:
    draft = _draft()

    result = mail.verify_and_dispatch(
        "a-different-draft", draft.confirmation_token, "s1", executor=lambda *a: {"sent": True}
    )

    assert result.success is False
    assert result.error_code == "UNAUTHORIZED_MUTATION"


def test_a_token_is_bound_to_its_session() -> None:
    draft = _draft()

    result = mail.verify_and_dispatch(
        "draft-1", draft.confirmation_token, "someone-elses-session", executor=lambda *a: {"sent": True}
    )

    assert result.success is False
    assert result.error_code == "UNAUTHORIZED_MUTATION"


def test_a_token_works_once() -> None:
    draft = _draft()

    first = mail.verify_and_dispatch("draft-1", draft.confirmation_token, "s1", executor=lambda *a: {"sent": True})
    second = mail.verify_and_dispatch("draft-1", draft.confirmation_token, "s1", executor=lambda *a: {"sent": True})

    assert first.success is True
    assert second.success is False
    assert "already used" in second.sanitized_output


def test_an_expired_token_is_refused() -> None:
    draft = _draft()

    result = mail.verify_and_dispatch(
        "draft-1",
        draft.confirmation_token,
        "s1",
        executor=lambda *a: {"sent": True},
        now=time.monotonic() + mail.TOKEN_TTL_SECONDS + 1,
    )

    assert result.success is False
    assert "expired" in result.sanitized_output


def test_the_send_tool_cannot_bypass_the_gate() -> None:
    """Even called through the ordinary path, a send with no draft is refused."""
    result = run(
        mail.MailToolCall("zoho_send_email", {"draft_id": "x", "confirmation_token": "abcdef"}),
        executor=_fake_client,
    )

    assert result.success is False
    assert result.error_code == "UNAUTHORIZED_MUTATION"


# ---------------------------------------------------------------- 4. the scrubber


@pytest.mark.parametrize(
    ("body", "secret"),
    [
        ("Your OTP is 483920 for the login. Do not share.", "483920"),
        ("OTP: 112233 is your verification code", "112233"),
        ("password: hunter2hunter2", "hunter2"),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----", "MIIEow"),
    ],
)
def test_a_secret_in_a_mail_body_never_reaches_the_model(body: str, secret: str) -> None:
    assert secret not in mail.sanitize_mail_body(body)


def test_a_tracking_pixel_and_an_inline_image_are_removed() -> None:
    body = (
        'Hello<img src="https://track.example.com/p.gif" width="1" height="1">'
        " data:image/png;base64," + "A" * 80
    )

    cleaned = mail.sanitize_mail_body(body)

    assert "track.example.com" not in cleaned
    assert "base64" not in cleaned


def test_an_ordinary_mail_is_left_readable() -> None:
    body = "Hi Asha,\n\nThe deploy finished at 4pm. Nothing else to report.\n\nThanks"

    assert mail.sanitize_mail_body(body) == body


def test_a_long_thread_is_truncated_with_a_notice() -> None:
    body = "line of a long thread\n" * 1000

    cleaned = mail.sanitize_mail_body(body)

    assert len(cleaned) < mail.MAX_BODY_CHARS + 100
    assert "[Thread truncated for context budget]" in cleaned


# ---------------------------------------------------------------- 5. failures


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "UNAUTHORIZED"), (404, "NOT_FOUND"), (429, "RATE_LIMITED"), (503, "UNAVAILABLE")],
)
def test_a_mail_failure_becomes_a_result_not_an_exception(status: int, code: str) -> None:
    class Boom(Exception):
        status_code = status

    async def failing(tool, arguments, token):
        raise Boom(f"HTTP {status}")

    result = run(mail.MailToolCall("zoho_search_emails", {"query": "invoice"}), executor=failing)

    assert result.success is False
    assert result.error_code == code
    assert result.error_message


def test_the_failure_speaks_the_users_language() -> None:
    class Boom(Exception):
        status_code = 401

    async def failing(tool, arguments, token):
        raise Boom("unauthorized")

    result = run(mail.MailToolCall("zoho_search_emails", {"query": "x"}), executor=failing, language="kn")

    assert result.error_message != mail._MESSAGES["UNAUTHORIZED"]["en"]


# ---------------------------------------------------------------- 6. speed


def test_the_policy_check_stays_under_its_budget() -> None:
    call = mail.MailToolCall("zoho_draft_email", {"to": ["a@b.com"], "subject": "s", "body": "b"})
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        mail.validate_arguments(call)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
