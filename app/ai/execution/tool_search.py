"""Find the tools that match a capability the model asked for.

The model is given a handful of tools up front plus ``find_tools``. When it needs
something else it asks, in English, and this returns the tools whose name,
description or argument names match. That is how a 41-tool catalogue costs a few
hundred tokens instead of 4,863 on every round of every message: the model pays only
for what it uses.

The match is lexical on purpose. It runs in-process, needs no model call, no network
and no embedding model in the deployment, and it cannot fail in a way that leaves the
model with nothing. The query comes from the model, which writes it in English whatever
language the user wrote in, so a Kannada request still finds an English description -
that is the whole reason the search does not have to be multilingual itself.

Ranking: a query word in a tool's name is worth three times the same word in its
description, because a name is what the tool is for and a description is only how it
says it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.ai.execution.tool_registry import EMAIL_TOOL_DECLARATIONS, TOOL_SPECS

DEFAULT_LIMIT = 5

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
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if word:
            words.append(word)
    return words


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
    for declaration in list(EMAIL_TOOL_DECLARATIONS) + list(extra_declarations):
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

    wanted = _tokens(query)
    if not wanted:
        return []
    scored: list[tuple[str, float]] = []
    for entry in _catalogue(extra_declarations):
        score = 0.0
        for token in wanted:
            if token in entry.name_tokens:
                score += 3.0
            elif token in entry.text_tokens:
                score += 1.0
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

    return [name for name, _ in score_tools(query, extra_declarations=extra_declarations)[: max(1, limit)]]


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
            "message": "Describe the capability you need in a short English query.",
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
