import re

import pytest

from app.ai.language_detection import (
    INDIC_CANCEL_WORDS,
    INDIC_CONFIRM_WORDS,
    INDIC_MAIL_WORDS,
    INDIC_NEGATION_WORDS,
    INDIC_SEND_WORDS,
    SCRIPT_RANGES,
    detect_response_language,
)


@pytest.mark.parametrize(
    ("message", "language"),
    [
        # Indian languages (native script, romanized, explicit names)
        ("ಕನ್ನಡ ಬಗ್ಗೆ ಹೇಳು", "Kannada"),
        ("kannada bagge helu", "Kannada"),
        ("Hindi me batao", "Hindi"),
        ("हिंदी में बताओ", "Hindi"),
        ("telugu lo cheppu", "Telugu"),
        ("తెలుగు లో చెప్పు", "Telugu"),
        ("tamil la sollu", "Tamil"),
        ("தமிழில் சொல்லு", "Tamil"),
        ("malayalam parayu", "Malayalam"),
        ("മലയാളത്തിൽ പറയൂ", "Malayalam"),
        ("marathi madhe sanga", "Marathi"),
        ("मराठीमध्ये सांगा", "Marathi"),
        ("bangla te bolo", "Bengali"),
        ("বাংলায় বলো", "Bengali"),
        ("assamese kobo", "Assamese"),
        ("অসমীয়াত কও", "Assamese"),
        ("gujarati ma kaho", "Gujarati"),
        ("ગુજરાતીમાં કહો", "Gujarati"),
        ("punjabi vich daso", "Punjabi"),
        ("ਪੰਜਾਬੀ ਵਿੱਚ ਦੱਸੋ", "Punjabi"),
        ("odia re kahantu", "Odia"),
        ("ଓଡ଼ିଆରେ କହନ୍ତୁ", "Odia"),
        ("urdu mein batao", "Urdu"),
        ("اردو میں بتائیں", "Urdu"),
        # Other languages Google supports (explicit names and native scripts)
        ("What is Indoone?", "English"),
        ("Answer in French please", "French"),
        ("en español por favor", "Spanish"),
        ("auf Deutsch bitte", "German"),
        ("日本語で答えて", "Japanese"),
        ("请用中文回答", "Chinese"),
        ("한국어로 답해줘", "Korean"),
        ("Ответь на русском", "Russian"),
        ("Γεια σου", "Greek"),
        ("שלום", "Hebrew"),
        ("สวัสดี", "Thai"),
        ("नमस्ते", "Hindi"),
    ],
)
def test_detect_response_language(message: str, language: str) -> None:
    assert detect_response_language(message) == language


def test_detect_response_language_always_returns_a_name() -> None:
    for message in ("", "   ", "12345", "!!!"):
        assert detect_response_language(message) == "English"


# ---------------------------------------------------------------------------
# The shared Indian-language word table.
#
# These are the lists every feature reads - the email confirmation, the cancellation,
# the send intent, the tool hints. A word typed into the wrong script would never match
# anything and would look exactly like a typo, so the script is checked rather than
# trusted.
# ---------------------------------------------------------------------------

# The names as app/ai/language_detection.py writes them. They are language names rather
# than script names, which is why Devanagari appears as Hindi and Gurmukhi as Punjabi.
# Assamese shares the Bengali script and Marathi shares Devanagari, so a word in either
# one is a word in both.
INDIAN_SCRIPTS = (
    "Hindi", "Bengali", "Punjabi", "Gujarati", "Odia",
    "Tamil", "Telugu", "Kannada", "Malayalam", "Urdu",
)

# One word per script, so a language that quietly lost its entry is caught. The words
# come from the table rather than being written out here: writing them by hand is how a
# test ends up asserting a code-point order nobody types, which is what happened when
# this was first written.
def test_every_shared_word_is_written_in_an_indian_script() -> None:
    """A word in the wrong script would never match, and would look like a typo."""
    lists = {
        "confirm": INDIC_CONFIRM_WORDS,
        "cancel": INDIC_CANCEL_WORDS,
        "send": INDIC_SEND_WORDS,
        "mail": INDIC_MAIL_WORDS,
        "negation": INDIC_NEGATION_WORDS,
    }
    for label, words in lists.items():
        for word in words:
            hits = [
                name
                for name, pattern in SCRIPT_RANGES
                if name in INDIAN_SCRIPTS and pattern.search(word)
            ]
            assert hits, f"{label}: {word!r} is not written in an Indian script"
            assert len(hits) == 1, f"{label}: {word!r} spans {hits}"
            assert not any(ch.isascii() and ch.isalpha() for ch in word), f"{label}: {word!r}"


def test_every_indian_script_has_a_confirmation_word() -> None:
    for script in INDIAN_SCRIPTS:
        pattern = next(p for name, p in SCRIPT_RANGES if name == script)
        assert any(
            pattern.search(word) for word in INDIC_CONFIRM_WORDS
        ), f"no confirmation word is written in {script}"


def test_every_indian_script_has_a_cancellation_word() -> None:
    for script in INDIAN_SCRIPTS:
        pattern = next(p for name, p in SCRIPT_RANGES if name == script)
        assert any(
            pattern.search(word) for word in INDIC_CANCEL_WORDS
        ), f"no cancellation word is written in {script}"


def test_every_indian_script_has_a_send_word() -> None:
    for script in INDIAN_SCRIPTS:
        pattern = next(p for name, p in SCRIPT_RANGES if name == script)
        assert any(
            pattern.search(word) for word in INDIC_SEND_WORDS
        ), f"no send word is written in {script}"


def _matches(words: tuple[str, ...], text: str) -> bool:
    """The same boundary rule the callers use, so the test cannot drift from them."""
    return any(re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) for word in words)


@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("ಇಮೇಲ್ ಕಳಿಸು", "Kannada"),
        ("ईमेल भेजो", "Hindi"),
        ("ইমেল পাঠাও", "Bengali"),
        ("ਈਮੇਲ ਭੇਜੋ", "Punjabi"),
        ("மின்னஞ்சல் அனுப்பு", "Tamil"),
        ("ఇమెయిల్ పంపు", "Telugu"),
        ("ഇമെയിൽ അയക്കൂ", "Malayalam"),
        ("ایمیل بھیجو", "Urdu"),
        ("ईमेल पाठवा", "Marathi"),
        ("ઈમેલ મોકલો", "Gujarati"),
        ("ଇମେଲ ପଠାଅ", "Odia"),
        ("ইমেইল পঠিয়াও", "Assamese"),
    ],
)
def test_an_everyday_email_request_is_recognised_in_every_language(
    message: str, language: str
) -> None:
    """People type the everyday spelling, not the bare dictionary word.

    The boundary rule the callers use means "ಇಮೇಲ್" does not match "ಮೇಲ್", so
    these sentences used to arrive as ordinary chat and never reached the email
    tools at all. This pins every language's everyday form.
    """
    assert _matches(INDIC_MAIL_WORDS, message), f"{language}: no mail word matched {message!r}"
    assert _matches(INDIC_SEND_WORDS, message), f"{language}: no send word matched {message!r}"
