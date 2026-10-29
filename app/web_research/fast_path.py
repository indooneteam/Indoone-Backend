"""Answer a web-research request in one turn, before the model can promise instead.

Reported from the app: "web research maadu jio ipo estakke listing agide" was answered with
"One moment, I will fetch the latest news" - and nothing was fetched. The user had to send
"kodu" to get an answer.

The cause is structural, not a bad model: search_web is a tool the model has to choose to
call. When it answers with a promise instead, the loop returns that text, the tool is never
run, and the promise becomes the reply. The user pays a second message for the answer the
first one should have carried.

So the search is run here, first, and the model is only asked to phrase what came back.
There is nothing left to promise.
"""

from __future__ import annotations

import logging
import re

from app.ai.gateway.service import generate_reply
from app.ai.quality.promise import is_promise
from app.ai.execution.text_match import subject_of
from app.web_research.grounding import GroundedEvidence, append_sources
from app.web_research.research import format_results
from app.web_research.tools import (
    SearchWebToolCall,
    execute_search_web_call,
    wants_web_research,
)

logger = logging.getLogger(__name__)

# The words that ask for the search rather than describe it. Stripped to leave the query.
_COMMAND_NOISE = re.compile(
    r"\b(web|internet|online|google|research|search|find|look\s*up|huduku|huduk|"
    r"maadu|madi|madu|kodu|helu|please|pls|for|about)\b"
    r"|ವೆಬ್|ಇಂಟರ್ನೆಟ್|ಜಾಲ|ಹುಡುಕು|ಹುಡುಕಿ|ಸಂಶೋಧನೆ|ಮಾಡು|ಮಾಡಿ|ಕೊಡು|ಹೇಳು|ಬಗ್ಗೆ",
    re.IGNORECASE,
)

# What the model is told once the results are in hand. Every clause here is a failure that
# was seen in the app: a promise, a question back, or an answer that ignored the sources.
_INSTRUCTION = (
    "A web search for this message has ALREADY been run, and its results are in the "
    "document context above. Answer from those results in this one reply, in the language "
    "the user wrote in. Do not say you will look it up, do not say 'one moment', do not ask "
    "the user to confirm before answering, and do not ask a question back. If the results "
    "do not contain the answer, say plainly what they do show and what they do not."
)

_NO_RESULTS = (
    "I searched, and no source came back that answers this. I will not guess at it: "
    "if you give me a narrower question I can search again."
)


# Grammar the sentence needs and the search does not: articles, and the Kannada case
# endings that arrive glued to a word ("ಜಾಲದಲ್ಲಿ", "ಬೆಂಗಳೂರಿನಲ್ಲಿ"). Removing them keeps the
# query to the thing being asked about.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "in", "on", "at", "to", "is", "are", "me", "my",
        # Kannada locatives that stand alone as a word, and what is left of them when
        # the noun before them was already taken out as command noise.
        "alli", "ralli", "nalli", "ದಲ್ಲಿ", "ಅಲ್ಲಿ", "ನಲ್ಲಿ", "ಲ್ಲಿ",
    }
)
_CASE_SUFFIXES = ("ದಲ್ಲಿ", "ನಲ್ಲಿ", "ಅಲ್ಲಿ", "ಯಲ್ಲಿ", "ಗೆ", "ಅನ್ನು", "ನ್ನು")


def _strip_case_suffix(word: str) -> str:
    for suffix in _CASE_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


def search_query_from(message: str) -> str:
    """The query inside a research request: the words that are not the command."""
    stripped = _COMMAND_NOISE.sub(" ", message or "")
    tokens = [
        _strip_case_suffix(token)
        for token in stripped.split()
        if token.casefold() not in _STOPWORDS
    ]
    query = " ".join(tokens).strip(" ?.!,")
    return query or " ".join((message or "").split()).strip()


async def web_research_answer(
    message: str,
    *,
    history: list[tuple[str, str]] | None = None,
    force: bool = False,
) -> str | None:
    """The grounded answer for a research request, or None when this is not one.

    ``force`` skips the gate: used when the model has already promised a lookup, which is
    its own statement that the question needs one.
    """
    if not force and not wants_web_research(message):
        return None

    query = search_query_from(message)

    if not query:
        return None

    try:
        result = await execute_search_web_call(
            SearchWebToolCall(query=query, model_content={})
        )
    except Exception as exc:  # a provider outage must not become a promise either
        logger.warning("web research failed: %s", exc)
        return (
            "I tried to search and the search itself failed. That is my side, not your "
            "question - ask me again in a moment."
        )

    if result.status == "invalid_query":
        return None

    if not result.results:
        return _NO_RESULTS

    evidence = [
        GroundedEvidence(
            title=str(item.title or ""),
            url=str(item.url or ""),
            snippet=str(item.snippet or ""),
        )
        for item in result.results
    ]

    try:
        answer = await generate_reply(
            message,
            history=history,
            document_context=format_results(list(result.results)),
            extra_system_instruction=_INSTRUCTION,
        )
    except Exception as exc:
        logger.warning("web research answer failed, returning the sources: %s", exc)
        answer = ""

    if not answer.strip():
        answer = format_results(list(result.results))

    try:
        return append_sources(answer, evidence)
    except Exception:
        # A quality check on the phrasing must not cost the user the answer.
        return answer


async def answer_instead_of_promise(
    reply: str,
    message: str,
    *,
    history: list[tuple[str, str]] | None = None,
) -> str | None:
    """The answer a promise was standing in for, or None when the reply is not a promise.

    A reply that names another subject - the mailbox, the thread, the repository - is left
    alone: those have deterministic routers, and forcing a web search over one would answer
    the wrong question.
    """
    if not is_promise(reply):
        return None

    if subject_of(message) is not None:
        return None

    return await web_research_answer(message, history=history, force=True)
