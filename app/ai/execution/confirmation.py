"""Whether a message confirms or cancels the action waiting on it.

A write that needs approval is held, and the next message from the user decides it. That
decision used to be read in two places, with two different rules, and both were wrong in
the same way: the GitHub one knew only English, romanised words and Kannada, so a Hindi,
Tamil, Telugu, Bengali, Malayalam or Punjabi "yes" was not recognised and the assistant
asked for permission again. The email one required the whole message to be the word, so
"ok send it" did not count, and any other message deleted the held action and handed the
turn to the model, which then said it was working on something it was not.

So the decision lives here, once:

* every language we answer in is recognised, in its own script and romanised;
* the word only has to start the message, because that is how people answer - "haan
  bhejo" is a yes;
* the numbered choices the prompt offers (1 to go ahead, 2 to cancel) are answers too,
  so a user who does not want to type a word can press a number.

Everything else is "other", and the caller must treat "other" as: the action is still
waiting, say so, and do not let anything else speak for it.
"""

from __future__ import annotations

import re
from typing import Final

from app.ai.language.language_detection import (
    INDIC_CANCEL_WORDS,
    INDIC_CONFIRM_WORDS,
    indic_pattern,
)

CONFIRM: Final = "confirm"
CANCEL: Final = "cancel"
OTHER: Final = "other"

# Words that are consent in this context but not general "yes" words elsewhere - "done,
# do it" and "go ahead" - kept local so the shared language lists stay about meaning.
_EXTRA_CONFIRM = (
    "ಆಯ್ತು", "ಆಗಲಿ", "ಮಾಡು", "ಮಾಡಿ", "ಕಳಿಸು",
    "आयतु", "करो", "कर दो", "भेज दो",
    "செய்", "பண்ணு",
    "చేయి", "చెయ్యి",
    "করো", "কর",
    "ചെയ്യൂ",
    "ਕਰੋ",
)

_ROMANISED_CONFIRM = (
    "ok", "okay", "okey", "k", "yes", "yep", "yeah", "ya", "sure", "confirm", "confirmed",
    "approve", "approved", "go ahead", "do it", "proceed", "send it", "haan", "han", "ha",
    "sari", "sare", "hudu", "howdu", "aaytu", "aytu", "aitu", "theek", "thik", "theek hai",
    "haan kar", "kar do", "karo", "maadu", "madi", "pannu", "chey", "cheyyu",
)
_ROMANISED_CANCEL = (
    "no", "nope", "cancel", "stop", "don't", "dont", "do not", "abort", "never mind",
    "beda", "beeda", "ille", "illa", "venda", "vaddu", "oddu", "nahi", "nahin", "na",
    "mat kar", "mat karo", "beda maadu",
)

# What may follow the word: the end of the message, or a separator. Not a word boundary -
# `\b` does not fire after an Indic word, because the last character of "ಸರಿ" or "हाँ" is a
# combining vowel sign or a candrabindu, and Python does not count those as word
# characters. Using `\b` here silently failed for every script we care about.
_END = r"(?=$|[\s,.!?;:।॥])"


def _alternation(words: tuple[str, ...]) -> str:
    # Longest first, so "haan" wins over "ha" and "okay" over "ok".
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


# The word has to lead the message, so "no problem, do it" is a cancel and "okay create
# maadu" is a confirm. A message that merely mentions a confirmation word in the middle is
# not an answer to the prompt, and stays "other".
_CONFIRM_RE = re.compile(
    r"^\s*(?:" + _alternation(_ROMANISED_CONFIRM + _EXTRA_CONFIRM) + "|" + indic_pattern(INDIC_CONFIRM_WORDS) + r")" + _END,
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^\s*(?:" + _alternation(_ROMANISED_CANCEL) + "|" + indic_pattern(INDIC_CANCEL_WORDS) + r")" + _END,
    re.IGNORECASE,
)
_NUMBER_CONFIRM_RE = re.compile(r"^\s*(?:1|1\.|one|option\s*1)\s*$", re.IGNORECASE)
_NUMBER_CANCEL_RE = re.compile(r"^\s*(?:2|2\.|two|option\s*2)\s*$", re.IGNORECASE)


# "no problem, do it" is consent, and its first word is a refusal. These idioms are
# stripped before the refusal is read, so the answer is what the user meant.
_HARMLESS_NO = re.compile(
    r"^\s*(?:no\s+(?:problem|worries|issue)|not\s+a\s+problem|no\s+worries)[\s,!.]*",
    re.IGNORECASE,
)


def classify_reply(message: str | None) -> str:
    """``confirm``, ``cancel`` or ``other`` for the message answering a held action."""

    text = _HARMLESS_NO.sub("", " ".join(str(message or "").split()).strip())
    if not text:
        return OTHER
    if _NUMBER_CONFIRM_RE.match(text):
        return CONFIRM
    if _NUMBER_CANCEL_RE.match(text):
        return CANCEL
    confirms = bool(_CONFIRM_RE.match(text))
    cancels = bool(_CANCEL_RE.match(text))
    # A message that leads with both - "no problem, do it" - is not an answer we can
    # read. Ask again rather than guess: cancelling something the user wanted is
    # annoying, and doing something they refused cannot be undone.
    if confirms and cancels:
        return OTHER
    if cancels:
        return CANCEL
    if confirms:
        return CONFIRM
    return OTHER


def is_confirm(message: str | None) -> bool:
    return classify_reply(message) == CONFIRM


def is_cancel(message: str | None) -> bool:
    return classify_reply(message) == CANCEL


def waiting_prompt(action: str = "the change") -> str:
    """What to say while an action is still waiting, in every language's terms.

    It names the state, offers a numbered choice so the answer does not depend on
    knowing a word in the assistant's language, and says plainly that nothing has
    happened yet - which is what the user is usually asking.
    """

    return (
        f"Nothing has happened yet: {action} is still waiting for your go-ahead.\n"
        "Reply 1 (or yes) to do it now, or 2 (or cancel) to drop it."
    )
