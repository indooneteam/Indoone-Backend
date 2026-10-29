"""How the fast-path routers read a message: words, not fragments.

Every router used to compare strings with `in`, which is a substring test. That is how
"hi" matched "Maha Shivaratri", how "a" matched "New Year's Day" and how the verb "check"
became a request for CI status. A substring test cannot tell a word from a piece of one.

So the comparison lives here, once:

* `words()` splits on anything that is not a letter or a digit, in Latin or Kannada script.
* `mentions()` is true when the phrase appears as whole words, or as the start of words
  that are at least `min_prefix` characters - "deep" finds "Deepavali", "hi" finds nothing.
* `is_pleasantry()` answers the other half of the problem: a message made only of greetings,
  acknowledgements and filler is never a request, whatever the rest of the pipeline thinks.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9\u0c80-\u0cff]+")

# Greetings, acknowledgements and filler, in the three ways people type them. A message
# built only from these is conversation, not a command.
_PLEASANTRIES = frozenset(
    {
        # English
        "hi", "hii", "hiii", "hello", "helo", "hey", "heyy", "yo", "sup", "hiya",
        "ok", "okay", "k", "kk", "yes", "yeah", "yep", "no", "nope", "nah",
        "thanks", "thank", "thx", "ty", "please", "pls", "sir", "madam", "bro", "dear",
        "good", "morning", "afternoon", "evening", "night", "day",
        "how", "are", "you", "whats", "what", "is", "up", "there", "doing",
        "help", "test", "again", "continue", "hmm", "hm", "oh", "ah", "wow", "cool",
        "nice", "great", "fine", "alright", "sure", "done", "welcome", "bye", "goodbye",
        "namaste", "namaskara", "namaskaram",
        # Kanglish
        "hai", "hlo", "hmm", "sari", "houdu", "haudu", "illa", "illa", "dhanyavada",
        "dhanyavaada", "namaskaragalu", "hegiddira", "hegiddiri", "chennagide",
        # Kannada script
        "ನಮಸ್ಕಾರ", "ನಮಸ್ಕಾರಗಳು", "ಹಾಯ್", "ಹಲೋ", "ಹೇಗಿದ್ದೀರ", "ಹೇಗಿದ್ದೀರಿ",
        "ಧನ್ಯವಾದ", "ಧನ್ಯವಾದಗಳು", "ಸರಿ", "ಹೌದು", "ಇಲ್ಲ", "ಒಕೆ", "ಚೆನ್ನಾಗಿದೆ",
        "ಶುಭೋದಯ", "ಗುಡ್", "ಮಾರ್ನಿಂಗ್",
    }
)


def words(text: str) -> list[str]:
    """The words in a message, lowercased, in Latin or Kannada script."""
    return _WORD.findall((text or "").casefold())


def is_pleasantry(text: str) -> bool:
    """True when a message is only greeting, acknowledgement or filler.

    Empty text is not a pleasantry - there is nothing to answer either way.
    """
    tokens = words(text)

    if not tokens:
        return False

    return all(token in _PLEASANTRIES for token in tokens)


def mentions(text: str, phrase: str, *, min_prefix: int = 4) -> bool:
    """True when ``phrase`` appears in ``text`` as words, or as the start of words.

    Whole words always count. A prefix counts only from ``min_prefix`` characters, so
    "deep" finds "Deepavali" while "di" finds nothing - the length rule exists because a
    two-letter prefix of a long name is not a mention of it.
    """
    wanted = words(phrase)
    found = words(text)

    if not wanted or not found:
        return False

    span = len(wanted)

    for index in range(len(found) - span + 1):
        window = found[index:index + span]
        matched = True

        for position, part in enumerate(window):
            target = wanted[position]

            if part == target:
                continue

            if len(target) >= min_prefix and part.startswith(target):
                continue

            matched = False
            break

        if matched:
            return True

    return False


# The subject a message is about, when it says so plainly. The festival calendar answers a
# question about a date; it must not answer a question about the mailbox, the conversation
# or the repository that merely mentions a festival. Measured before this existed:
# "what did we discuss about deepavali" and "unread emails about deepavali" both came back
# with the holiday calendar.
_SUBJECTS = {
    "email": re.compile(
        r"\b(emails?|mails?|inbox|gmail|zoho|outlook|unread)\b|ಮೇಲ್|ಇನ್‌ಬಾಕ್ಸ್|ಮೈಲ್",
        re.IGNORECASE,
    ),
    "conversation": re.compile(
        r"\b(discuss(ed)?|conversations?|chats?|thread|history|summar(y|ise|ize)|talked|we\s+said|"
        r"maatu|maatadu|matu|maathaadid)\b|ಮಾತು|ಸಂಭಾಷಣೆ|ಚಾಟ್|ಸಾರಾಂಶ|ಮಾತಾಡಿದ",
        re.IGNORECASE,
    ),
    "code": re.compile(
        r"\b(repos?|repositor(y|ies)|branch(es)?|commits?|pull\s*requests?|prs?|issues?|github|"
        r"workflows?|pipelines?)\b|ರೆಪೊ|ಶಾಖೆ|ಕಮಿಟ್",
        re.IGNORECASE,
    ),
}


def subject_of(text: str) -> str | None:
    """The subject a message names, or None when it names none of them.

    Deliberately one answer, not a set: a router that answers dates asks this and stands
    down for anything that is plainly about another subject.
    """
    cleaned = (text or "").strip()

    if not cleaned:
        return None

    for name, pattern in _SUBJECTS.items():
        if pattern.search(cleaned):
            return name

    return None
