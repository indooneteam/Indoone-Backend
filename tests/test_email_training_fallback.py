from __future__ import annotations

import pytest

from app.ai import service
from app.ai.answer_quality import user_safe_failure
from app.ai.email_training_fallback import (
    has_email_payload,
    needs_email_payload,
)


def test_email_payload_detection() -> None:
    prompt = (
        "Analyze this email.\n"
        "Subject: Verify account\n"
        "From: Security Team <notice@example.com>\n"
        "To: User <user@example.com>\n"
        "Your account will close soon. Send your password within 15 minutes."
    )
    assert has_email_payload(prompt)


def test_generic_email_phishing_question_requests_email_content() -> None:
    assert needs_email_payload("Is this email phishing?")


@pytest.mark.asyncio
async def test_email_safe_failure_uses_training_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        async def answer(self, message, *, history=None, document_context=""):
            return user_safe_failure()

    monkeypatch.setattr(service, "UniversalQuestionAnswerPipeline", lambda *args, **kwargs: FakePipeline())

    prompt = (
        "Analyze this email.\n"
        "Subject: Verify account\n"
        "From: Security Team <notice@example.com>\n"
        "To: User <user@example.com>\n"
        "Your account will close soon. Send your password within 15 minutes."
    )

    reply = await service.generate_reply(prompt)

    assert reply != user_safe_failure()
    assert "Classification:" in reply or "phishing" in reply.casefold()


@pytest.mark.asyncio
async def test_generic_email_question_does_not_use_safe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePipeline:
        async def answer(self, message, *, history=None, document_context=""):
            return user_safe_failure()

    monkeypatch.setattr(service, "UniversalQuestionAnswerPipeline", lambda *args, **kwargs: FakePipeline())

    reply = await service.generate_reply("Is this email phishing?")

    assert "Paste the email content" in reply
