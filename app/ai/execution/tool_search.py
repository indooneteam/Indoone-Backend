"""Find the tools that match a capability the model asked for.

The model is given a handful of tools up front plus ``find_tools``. When it needs
something else it asks and this returns the tools whose name, description or
argument names match. That is how a 41-tool catalogue costs a few hundred tokens
instead of 4,863 on every round of every message: the model pays only for what it
uses.

The match is lexical on purpose. It runs in-process, needs no model call, no network
and no embedding model in the deployment, and it cannot fail in a way that leaves the
model with nothing.

Language: the catalogue is written in English, and the model usually writes its query
in English too. But when the user asks in Kannada, the model sometimes passes the
user's own words straight through, and those words used to tokenise to nothing - a
query of "ಬ್ರಾಂಚ್ ಎಷ್ಟಿದೆ" produced an empty token list and ``find_tools`` answered
"no tool matches that", which the model then reported as having no access at all.
``_CAPABILITY_ALIASES`` closes that hole: each Kannada or romanised-Kannada
("Kanglish") surface word maps to the English capability words the catalogue uses.
It is a small explicit vocabulary rather than a translation model - it is auditable,
it needs nothing at runtime, and a word that is missing degrades to the previous
behaviour instead of inventing a match.

Ranking: a query word in a tool's name is worth three times the same word in its
description, because a name is what the tool is for and a description is only how it
says it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.ai.execution.tool_registry import EMAIL_TOOL_DECLARATIONS, SEARCH_WEB_DECLARATION, TOOL_SPECS

DEFAULT_LIMIT = 5

# A query scores the sum of the weights of the words it matches. Below this it is not
# a tool request at all - "explain what a pull request is" shares words with five
# GitHub tools, but only the common ones, and the sum stays under the line. Tuned on
# data/eval/tool_selection.jsonl; the gate in the tests keeps it honest.
MIN_SCORE = 4.0

# A word that reached the query through ``_CAPABILITY_ALIASES`` is worth more than an
# ordinary word. "ಬ್ರಾಂಚ್ ಎಷ್ಟಿದೆ" is two Kannada words and one capability, and after
# expansion it is a single English token - the same shape as a bare "branch", which
# scores under MIN_SCORE on its own. The table is curated, so the match is not a guess
# and must not be discarded for being short. English queries never take this path.
_ALIAS_WEIGHT = 1.5

# Words that carry no capability. Kept short and English-only: the query is written by
# the model, and it writes English, so this list does not have to cover a language.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "get", "how", "i", "in", "into", "is", "it", "me", "my", "of",
        "on", "or", "please", "some", "that", "the", "their", "them", "then", "this",
        "to", "tool", "tools", "use", "user", "want", "was", "what", "when", "where",
        "which", "who", "will", "with", "you", "your",
    }
)

_WORD = re.compile(r"[a-z0-9]+")
# Any run of characters that is not whitespace and not ASCII punctuation. Written this
# way rather than as ``[^\W_]+`` because Python's ``\w`` does not include combining
# marks: a pattern built on it splits a Kannada word at every vowel sign, so
# "ಬ್ರಾಂಚ್" arrived as "ಬ", "ರ", "ಚ" and matched nothing.
_UNICODE_WORD = re.compile(
    "[^\\s\\u0000-\\u002f\\u003a-\\u0040\\u005b-\\u0060\\u007b-\\u007f]+"
)

# Surface word -> the English capability words the catalogue is written in.
#
# Two kinds of key, because users type both: Kannada script (ಬ್ರಾಂಚ್) and romanised
# Kannada ("branch eshtu ide"). Keys are compared casefolded. Values are the words
# that actually appear in tool names and descriptions, never a paraphrase of them.
_CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    # Kannada script.
    "ಬ್ರಾಂಚ್": ("branch",),
    "ಬ್ರಾಂಚ್‌ಗಳು": ("branch",),
    "ರೆಪೊ": ("repository", "repo"),
    "ರೆಪೋ": ("repository", "repo"),
    "ರಿಪೊ": ("repository", "repo"),
    "ರೆಪೊಸಿಟರಿ": ("repository",),
    "ಇಮೇಲ್": ("email",),
    "ಮೇಲ್": ("email", "mail"),
    "ಫೈಲ್": ("file",),
    "ಫೈಲ್‌ಗಳು": ("file",),
    "ಕಮಿಟ್": ("commit",),
    "ಪುಲ್": ("pull",),
    "ರಿಕ್ವೆಸ್ಟ್": ("request",),
    "ಇಷ್ಯೂ": ("issue",),
    "ಇಶ್ಯೂ": ("issue",),
    "ಕೋಡ್": ("code",),
    "ರಿವ್ಯೂ": ("review",),
    "ಹುಡುಕು": ("search",),
    "ಹುಡುಕಿ": ("search",),
    "ಕಳಿಸು": ("send",),
    "ಕಳುಹಿಸು": ("send",),
    "ತೋರಿಸು": ("list", "show"),
    "ತೋರಿಸಿ": ("list", "show"),
    "ಎಷ್ಟು": ("count", "list"),
    "ಎಲ್ಲಾ": ("list", "all"),
    "ಪಟ್ಟಿ": ("list",),
    "ರಚಿಸು": ("create",),
    "ಕ್ರಿಯೇಟ್": ("create",),
    "ಮಾಡು": ("create", "do"),
    "ಡಿಲೀಟ್": ("delete",),
    "ಅಳಿಸು": ("delete",),
    "ಮರ್ಜ್": ("merge",),
    "ಟೆಸ್ಟ್": ("test",),
    "ರನ್": ("run",),
    "ಡಿಪ್ಲಾಯ್": ("deploy",),
    "ಬಗ್": ("bug", "issue"),
    "ಟಾಸ್ಕ್": ("task",),
    "ಜಿರಾ": ("jira",),
    "ಸ್ಲ್ಯಾಕ್": ("slack",),
    "ಡಾಕ್ಯುಮೆಂಟ್": ("document",),
    "ಶೀಟ್": ("sheet",),
    "ಕ್ಯಾಲೆಂಡರ್": ("calendar",),
    "ಮೀಟಿಂಗ್": ("meeting",),
    "ವೆಬ್": ("web",),
    "ಸುದ್ದಿ": ("news",),
    "ಸಾರಾಂಶ": ("summary", "summarize"),
    "ಅನುವಾದ": ("translate",),
    "ಲೆಕ್ಕ": ("calculate",),
    "ಗಣನೆ": ("calculate",),
    "ಹೆಸರು": ("name",),
    "ಸ್ಟೇಟಸ್": ("status",),
    "ಚೆಕ್": ("check",),
    "ಓದು": ("read",),
    "ಓದಿ": ("read",),
    "ಬರೆ": ("write",),
    "ಕಾಮೆಂಟ್": ("comment",),
    "ಅಪ್ರೂವ್": ("approve",),
    "ರಿಲೀಸ್": ("release",),
    "ವರ್ಕ್ಫ್ಲೋ": ("workflow",),
    "ರಿಪೋರ್ಟ್": ("report",),
    "ಪ್ರೊಫೈಲ್": ("profile",),
    "ಚಿತ್ರ": ("image", "media"),
    "ಫೋಟೋ": ("image", "media"),
    "ವಿಡಿಯೋ": ("video", "media"),
    "ಧ್ವನಿ": ("voice", "audio"),
    "ಸಂಖ್ಯೆ": ("number", "count"),
    # Romanised Kannada / Kanglish. Only words that name a capability: a verb like
    # "maadu" is included because it is what a user writes around the noun, but a
    # filler word is not, because a filler must never raise a score.
    "eshtu": ("count", "list"),
    "estu": ("count", "list"),
    "helu": ("tell",),
    "torsu": ("list", "show"),
    "torisu": ("list", "show"),
    "maadu": ("create", "do"),
    "madu": ("create", "do"),
    "beku": ("want",),
    "nodu": ("show",),
    "huduk": ("search",),
    "huduku": ("search",),
    "kalisi": ("send",),
    "kalisu": ("send",),
    "ella": ("list", "all"),
    "listu": ("list",),
    "hesaru": ("name",),
    "bare": ("write",),
    "bari": ("write",),
    "odhu": ("read",),
    "odi": ("read",),
    "alisu": ("delete",),
    "madide": ("do",),
    "aagide": ("do",),
}


@dataclass(frozen=True)
class _Entry:
    name: str
    description: str
    name_tokens: frozenset[str]
    text_tokens: frozenset[str]


def _tokens(text: str) -> list[str]:
    """Lowercase words, stopwords removed, plurals folded to one form."""

    words: list[str] = []
    for word in _WORD.findall(str(text or "").lower()):
        if word in _STOPWORDS:
            continue
        # "branches" folded to "branche" under the plain plural rule, so a query of
        # "branch" never matched the tool named github_list_branches - the one tool
        # whose name is exactly the word being asked about. These endings fold to the
        # stem the singular already has.
        if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        # analyse / analyze, summarise / summarize: the user's spelling and the tool's
        # description should not have to agree for a search to work.
        if len(word) > 5 and word.endswith("ise"):
            word = word[:-3] + "ize"
        elif len(word) > 5 and word.endswith("yse"):
            word = word[:-3] + "yze"
        if word:
            words.append(word)
    return words


def _query_tokens(text: str) -> tuple[list[str], frozenset[str]]:
    """The English capability words a query names, and which of them came from an alias.

    Each surface word is looked up in ``_CAPABILITY_ALIASES`` first, so a Kannada or
    Kanglish word contributes the English words the catalogue uses. A word that is
    neither in the table nor ASCII contributes nothing - it cannot match an English
    description - and an ASCII word is kept exactly as it is, so an English query
    scores today exactly as it scored before this table existed.

    The alias words are returned separately because they are not a guess: the table is
    a curated statement that this word names this capability, so a match on one is
    stronger evidence than a match on an ordinary word, and it is weighted that way in
    ``score_tools``.
    """

    expanded: list[str] = []
    from_alias: set[str] = set()
    for raw in _UNICODE_WORD.findall(str(text or "")):
        word = raw.casefold()
        aliases = _CAPABILITY_ALIASES.get(word)
        if aliases is not None:
            expanded.extend(aliases)
            from_alias.update(aliases)
        elif word.isascii():
            expanded.append(word)
    return _tokens(" ".join(expanded)), frozenset(from_alias)


def _query_terms(text: str) -> list[str]:
    """The English capability words a query names, ignoring where they came from."""

    return _query_tokens(text)[0]


def _entry(name: str, description: str, parameters: Any) -> _Entry:
    """One searchable tool: its name, its description, and its argument names."""

    argument_words: list[str] = []
    if isinstance(parameters, dict):
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            for key, value in properties.items():
                argument_words.append(str(key).replace("_", " "))
                if isinstance(value, dict) and value.get("description"):
                    argument_words.append(str(value["description"]))
    text = " ".join([description or "", " ".join(argument_words)])
    return _Entry(
        name=name,
        description=str(description or ""),
        name_tokens=frozenset(_tokens(name.replace("_", " "))),
        text_tokens=frozenset(_tokens(text)),
    )


def _catalogue(extra_declarations: Iterable[dict[str, Any]] = ()) -> list[_Entry]:
    """Every tool that can be discovered, from the one place tools are defined."""

    entries = [
        _entry(spec.name, spec.description, spec.parameters)
        for spec in TOOL_SPECS
        if getattr(spec, "model_facing", True)
    ]
    declarations = [SEARCH_WEB_DECLARATION] + list(EMAIL_TOOL_DECLARATIONS) + list(extra_declarations)
    for declaration in declarations:
        name = str(declaration.get("name") or "").strip()
        if name:
            entries.append(
                _entry(name, str(declaration.get("description") or ""), declaration.get("parameters"))
            )
    return entries


def score_tools(
    query: str,
    *,
    extra_declarations: Iterable[dict[str, Any]] = (),
) -> list[tuple[str, float]]:
    """Every tool that matches the query, best first, with its score."""

    wanted, alias_terms = _query_tokens(query)
    if not wanted:
        return []
    catalogue = _catalogue(extra_declarations)
    total = len(catalogue) or 1
    # A word that appears in most descriptions separates nothing, so it is worth
    # little; a word only one tool uses is worth a lot. Without this, "pull" and
    # "request" make "explain what a pull request is" look like a GitHub request.
    weights: dict[str, float] = {}
    for token in wanted:
        document_frequency = sum(
            1 for entry in catalogue if token in entry.name_tokens or token in entry.text_tokens
        )
        weights[token] = math.log(1.0 + total / (1.0 + document_frequency))
    scored: list[tuple[str, float]] = []
    for entry in catalogue:
        score = 0.0
        for token in wanted:
            weight = weights[token] * (_ALIAS_WEIGHT if token in alias_terms else 1.0)
            if token in entry.name_tokens:
                score += 2.0 * weight
            elif token in entry.text_tokens:
                score += weight
        if score > 0:
            scored.append((entry.name, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def search_tool_names(
    query: str,
    *,
    extra_declarations: Iterable[dict[str, Any]] = (),
    limit: int = DEFAULT_LIMIT,
) -> list[str]:
    """The names of the tools worth offering for this query, best first."""

    return [
        name
        for name, score in score_tools(query, extra_declarations=extra_declarations)[: max(1, limit)]
        if score >= MIN_SCORE
    ]


def tool_description(name: str, extra_declarations: Iterable[dict[str, Any]] = ()) -> str:
    """The one-line description of a tool, for the result the model reads."""

    for entry in _catalogue(extra_declarations):
        if entry.name == name:
            return entry.description
    return ""


def find_tools_payload(query: str, *, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """What ``find_tools`` returns, in the shape the model reads.

    Nothing is filtered by what the user has connected. A tool whose connector is not
    connected is still offered, because its executor answers with "connect it in
    Plugins" - which is a better reply than the model never knowing the capability
    exists and saying it cannot help.
    """

    cleaned = " ".join(str(query or "").split()).strip()
    if not cleaned:
        return {
            "ok": False,
            "code": "EMPTY_QUERY",
            "message": "Describe the capability you need in a short query, in English or Kannada.",
        }
    names = search_tool_names(cleaned, limit=limit)
    if not names:
        return {
            "ok": True,
            "result": {
                "tools": [],
                "note": "No tool matches that. Answer from what you know, or say plainly that you cannot do it.",
            },
        }
    return {
        "ok": True,
        "result": {
            "tools": [{"name": name, "description": tool_description(name)} for name in names]
        },
    }
