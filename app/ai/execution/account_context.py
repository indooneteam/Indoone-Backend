"""What the model is told about the accounts this user has connected.

Two failures come from this being missing, and both look like the assistant lying:

* It answered "I don't have access to your GitHub" while GitHub was connected. The tools
  were there; the model had no idea an account was connected, so it reached for its
  default safety answer.
* It kept saying it after the account was connected, because the earlier refusal was
  still in the last twelve messages it was sent. A model reads its own last answer as
  context, and an old "I can't" is a strong pull.

So the connected set goes into the instruction, and the stale refusals come out of the
history. Both are about the same thing: the model should be reasoning from the state we
have now, not from the state a previous turn was in.
"""

from __future__ import annotations

from typing import Iterable

from app.ai.execution.tool_registry import TOOL_SPECS
from app.capabilities.connectors.store import get_integration_token_metadata

# The email tools live in the email plugin and are gated on the mailbox connector, so it
# is named here rather than read from a spec.
_EXTRA_CONNECTORS = ("zoho_mail",)

# Sentences a refusal is made of. Used only to drop an old assistant turn from the
# context, never to judge a new answer.
_REFUSAL_MARKERS = (
    "don't have access",
    "do not have access",
    "can't access",
    "cannot access",
    "not connected to this indoone account",
    "connect it from plugins",
    "connect it in plugins",
)


def connector_ids() -> tuple[str, ...]:
    """Every connector a user can connect, in one list."""

    ids = {spec.connector_id for spec in TOOL_SPECS if getattr(spec, "connector_id", None)}
    return tuple(sorted(ids | set(_EXTRA_CONNECTORS)))


def connected_accounts(user_id: str) -> list[str]:
    """The connectors this user has actually connected, in a stable order."""

    normalized = str(user_id or "").strip()
    if not normalized:
        return []
    connected: list[str] = []
    for connector_id in connector_ids():
        try:
            if get_integration_token_metadata(normalized, connector_id) is not None:
                connected.append(connector_id)
        except Exception:
            # A store that cannot answer must not break the request; the model simply
            # gets no line about accounts.
            continue
    return connected


def accounts_instruction(user_id: str) -> str:
    """The line to add to the instruction, or empty when nothing is connected."""

    connected = connected_accounts(user_id)
    if not connected:
        return ""
    names = ", ".join(connected)
    return (
        f"The user has connected these accounts: {names}. You can use their tools now. "
        "Never say you cannot access an account that is listed here - call its tool. If a "
        "tool reports the account is not connected, say so and point to Plugins."
    )


def without_stale_refusals(
    history: Iterable[tuple[str, str]] | None,
    user_id: str,
) -> list[tuple[str, str]] | None:
    """Drop earlier assistant turns that claimed it had no access.

    Only when something is connected: with nothing connected, a refusal was accurate and
    is worth keeping. The turns dropped are the assistant's own, never the user's.
    """

    if history is None:
        return None
    turns = list(history)
    if not connected_accounts(user_id):
        return turns
    kept: list[tuple[str, str]] = []
    for role, text in turns:
        lowered = str(text or "").casefold()
        if str(role) == "assistant" and any(marker in lowered for marker in _REFUSAL_MARKERS):
            continue
        kept.append((role, text))
    return kept
