"""Whether a message confirms or cancels the action waiting on it.

This is the decision that made the assistant ask for permission again and again. The rule
used to be written twice, in two different shapes, and both shapes only knew English,
romanised words and - in one of them - Kannada. A Hindi, Tamil, Telugu, Bengali, Malayalam
or Punjabi yes did not register, so the same question came back, and the held action was
never carried out.

Every case below is a message a real user sent or would send. The gate is not the wording
but the answer: a yes in any language we support must be a yes.
"""

from __future__ import annotations

import pytest

from app.ai.execution.confirmation import CANCEL, CONFIRM, OTHER, classify_reply


@pytest.mark.parametrize(
    "message",
    [
        # English and romanised
        "ok", "okay", "okay create maadu", "yes", "yep", "sure", "go ahead", "do it",
        "confirm", "no problem, do it", "no worries, send it",
        # romanised Kannada / Hindi
        "haan", "haan bhejo", "sari", "hudu", "aaytu", "maadu", "theek hai",
        # the scripts
        "ಸರಿ", "ಆಯ್ತು", "ಹೌದು ಕಳಿಸು", "ಆಯ್ತು ಮಾಡು",
        "हाँ", "हाँ भेजो", "ठीक है",
        "ஆம்", "சரி", "அனுப்பு",
        "అవును", "సరే", "పంపు",
        "হ্যাঁ", "ঠিক",
        "അതെ", "ശരി",
        "ਹਾਂ", "ਠੀਕ",
        # the numbered choice the prompt offers
        "1", "1.", "one",
    ],
)
def test_a_yes_is_a_yes_in_every_language(message: str) -> None:
    assert classify_reply(message) == CONFIRM


@pytest.mark.parametrize(
    "message",
    [
        "no", "nope", "cancel", "stop", "don't", "do not", "never mind",
        "beda", "ille", "illa", "venda", "nahi",
        "ಬೇಡ", "ನಿಲ್ಲಿಸು", "नहीं", "रद्द", "இல்லை", "వద్దు", "না", "അല്ല", "ਨਹੀਂ",
        "2", "2.", "two",
    ],
)
def test_a_no_is_a_no_in_every_language(message: str) -> None:
    assert classify_reply(message) == CANCEL


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
        "what is the weather",
        "did it happen",
        "ಏನಾಯ್ತು",
        "wait what happened",
        "please confirm later",
        "no problem",
        "why did you not do it",
    ],
)
def test_everything_else_is_not_an_answer(message: str) -> None:
    """A question is not consent, and it is not a refusal either.

    The caller must hold the action and say so - which is what stops the assistant
    saying it is working on something it is not.
    """
    assert classify_reply(message) == OTHER


def test_a_refusal_wins_when_the_message_leads_with_both() -> None:
    """Doing something the user refused cannot be undone, so it is never guessed."""
    assert classify_reply("beda, but ok later") == CANCEL


def test_the_word_has_to_lead_the_message() -> None:
    assert classify_reply("can you confirm that later") == OTHER
    assert classify_reply("okay") == CONFIRM


def test_the_waiting_prompt_names_the_state_and_offers_a_number() -> None:
    from app.ai.execution.confirmation import waiting_prompt

    prompt = waiting_prompt("the email to ravi")

    assert "still waiting" in prompt
    assert "Nothing has happened yet" in prompt
    assert "Reply 1" in prompt and "2" in prompt
