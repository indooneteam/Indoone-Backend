"""Pan-Indic alignment across the agents: digits, holidays, folds, pronouns, glyphs.

Each of these was a real miss with the same shape: the code worked in English and in
Romanised text, and silently did nothing when the same thing arrived in an Indian script.
"""

from __future__ import annotations

import unicodedata

import pytest

from app.agents import load

vault = load("03_vault_guardian")
intent = load("05_intent_purity_extractor")
treasury = load("07_treasury_cache")
condenser = load("08_context_condenser")
arbiter = load("10_model_arbiter")


# ---------------------------------------------------------------- 1. agent 03: digits


@pytest.mark.parametrize(
    ("script", "digits"),
    [
        ("Kannada", "೯೮೭೬೫೪೩೨೧೦"),
        ("Devanagari", "९८७६५४३२१०"),
        ("Telugu", "౯౮౭౬౫౪౩౨౧౦"),
        ("Tamil", "௯௮௭௬௫௪௩௨௧௦"),
        ("Bengali", "৯৮৭৬৫৪৩২১০"),
        ("Gujarati", "૯૮૭૬૫૪૩૨૧૦"),
        ("Malayalam", "൯൮൭൬൫൪൩൨൧൦"),
    ],
)
def test_a_number_in_a_native_script_is_redacted(script: str, digits: str) -> None:
    """A phone number typed in Kannada is the same phone number."""
    verdict = vault.redact(f"my phone {digits}")

    assert digits not in verdict.redacted_prompt, f"{script} digits reached the provider"
    assert "Redacted" in verdict.redacted_prompt


def test_ascii_numbers_still_work() -> None:
    assert "9876543210" not in vault.redact("my phone 9876543210").redacted_prompt


def test_an_ordinary_number_is_left_alone() -> None:
    """Digits normalised, but not everything with digits becomes a phone number."""
    assert "2026" in vault.redact("the festival is in 2026").redacted_prompt


# ---------------------------------------------------------------- 2. agent 05: fast path


@pytest.mark.parametrize(
    ("script", "question"),
    [
        ("Kannada", "ನಾಳೆ ರಜಾ ಇದೆಯಾ"),
        ("Hindi", "कल छुट्टी है क्या"),
        ("Telugu", "రేపు సెలవా"),
        ("Tamil", "நாளை விடுமுறையா"),
        ("Bengali", "কাল কি ছুটি"),
        ("Gujarati", "આજે રજા છે"),
        ("Malayalam", "ഇന്ന് അവധി ഉണ്ടോ"),
    ],
)
def test_a_native_holiday_question_reaches_the_calendar(script: str, question: str) -> None:
    """0 tokens: the calendar answers this, and it always should have."""
    verdict = intent.extract(question)

    assert verdict.action == "CHECK_HOLIDAY", f"{script} went to the model"
    assert verdict.resource == "HOLIDAY_CALENDAR"
    assert verdict.is_fast_path is True
    assert verdict.fast_path_target == "calendar"


@pytest.mark.parametrize(
    "question",
    [
        "write a poem about the sea",
        "merge the pr",
        "commit 632a7f6d09",
        "explain how the scheduler works in detail please",
    ],
)
def test_a_native_fast_path_does_not_widen_the_english_one(question: str) -> None:
    assert intent.extract(question).is_fast_path is False


def test_a_postposition_is_peeled_from_an_entity() -> None:
    assert intent.peel_postposition("branchನಲ್ಲಿ") == "branch"
    assert intent.peel_postposition("में") == "में", "a bare postposition is not a word"


# ---------------------------------------------------------------- 3. agent 07: folding


@pytest.mark.parametrize(
    ("script_form", "roman_form"),
    [
        ("कल छुट्टी है क्या", "kal chutti hai kya"),
        ("ನಾಳೆ ರಜೆ ಇದೆಯಾ", "naale raje ideya"),
        ("రేపు సెలవు", "repu selavu"),
    ],
)
def test_one_question_one_key(script_form: str, roman_form: str) -> None:
    """The same question typed two ways must not be cached twice."""
    language = "hi"

    assert treasury.canonical_key(script_form, language, "CASUAL") == treasury.canonical_key(
        roman_form, language, "CASUAL"
    )


def test_folding_does_not_erase_the_language() -> None:
    """Same fact, two answers: the language stays part of the key."""
    assert treasury.canonical_key("when is deepavali", "en", "CASUAL") != treasury.canonical_key(
        "when is deepavali", "kn", "CASUAL"
    )


def test_folding_does_not_erase_the_register() -> None:
    assert treasury.canonical_key("when is deepavali", "en", "CASUAL") != treasury.canonical_key(
        "when is deepavali", "en", "HONORIFIC"
    )


def test_two_different_holidays_stay_different() -> None:
    """Folding is by concept, not by topic."""
    assert treasury.canonical_key("when is deepavali", "en", "CASUAL") != treasury.canonical_key(
        "when is ugadi", "en", "CASUAL"
    )


# ---------------------------------------------------------------- 4. agent 08: pronouns


HISTORY = [
    {"role": "user", "content": "show branch feat/auth"},
    {"role": "assistant", "content": "branch feat/auth has two commits"},
]


@pytest.mark.parametrize(
    ("script", "follow_up"),
    [
        ("Kannada", "ಅದರ diff ತೋರಿಸು"),
        ("Hindi", "उसका diff दिखाओ"),
        ("Telugu", "దాని diff చూపించు"),
        ("Tamil", "அதன் diff காட்டு"),
        ("Bengali", "তার diff দেখাও"),
        ("Malayalam", "അതിന്റെ diff കാണിക്കുക"),
    ],
)
def test_a_native_pronoun_keeps_the_context(script: str, follow_up: str) -> None:
    """A language shift does not restart the thread."""
    verdict = condenser.condense(follow_up, HISTORY)

    assert verdict.requires_history is True, f"{script} lost its context"
    assert verdict.dependency_type == condenser.ANAPHORA_RESOLVED
    assert verdict.resolved_target is not None


# ---------------------------------------------------------------- 5. agent 10: glyphs


@pytest.mark.parametrize("word", ["ರಾಷ್ಟ್ರೀಯ", "दृष्टिकोण", "ವಿದ್ಯಾರ್ಥಿ"])
def test_a_truncated_indic_text_keeps_its_glyphs(word: str) -> None:
    """A cut inside a cluster is what shows as a box on the user's screen."""
    body = (word + " ") * 400

    text = arbiter.synthesize(body, "instagram").formatted_text
    trimmed = text[:-1] if text.endswith("…") else text  # drop our own ellipsis

    assert len(text) <= arbiter.INSTAGRAM_LIMIT
    assert trimmed, "something must survive"

    # nothing may begin a cluster: no leading combining mark, no trailing virama
    assert not unicodedata.combining(trimmed[0])
    assert trimmed[-1] not in arbiter._VIRAMA


def test_whatsapp_chunks_never_split_a_cluster() -> None:
    body = "ರಾಷ್ಟ್ರೀಯ " * 1000

    chunks = arbiter.synthesize(body, "whatsapp").channel_payload["chunks"]

    assert all(chunk and not unicodedata.combining(chunk[0]) for chunk in chunks)
    assert all(chunk[-1] not in arbiter._VIRAMA for chunk in chunks)


def test_a_short_indic_text_is_untouched() -> None:
    body = "ರಾಷ್ಟ್ರೀಯ ಹಬ್ಬ"

    assert arbiter.synthesize(body, "instagram").formatted_text == body
