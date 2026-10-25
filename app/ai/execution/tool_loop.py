"""One agent loop: the model chooses any tool, the backend runs it.

Before this module, ``/chat`` tried a fixed order of plugins (email, then
GitHub, then a plain answer) and each plugin decided for itself — with
hand-written regex — whether it applied. Here there is a single list of tools
(``model_tool_declarations``) and a single dispatcher (``dispatch_tool``), so
the model itself decides whether a tool is needed and which one to call, in any
language, exactly like a general assistant.

Tools that change data (``requires_approval`` in the registry) are offered to
the model too, but the dispatcher refuses to run one until the caller presents
a server-issued approval token for that exact tool (see ``app.ai.execution.approval``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable

from app.ai.execution.approval import decide_tool
from app.ai.gateway.ai_rules import (
    APPROVAL_REQUIRED,
    SEARCH_FAILED,
    SEARCH_RESULT_MESSAGES,
    TOOL_SYSTEM_INSTRUCTION as _UNIFIED_SYSTEM_INSTRUCTION,
    UNKNOWN_TOOL,
)
from app.ai.execution.tool_registry import (
    EMAIL_TOOL_DECLARATIONS,
    SEARCH_WEB_DECLARATION,
    GITHUB_TOOL_DECLARATIONS,
    TOOL_SPECS,
    get_tool_spec,
    tool_declaration,
)
from app.ai.execution.tool_search import find_tools_payload
from app.ai.execution.tools import run_tool_async

_EMAIL_TOOL_NAMES = frozenset(item["name"] for item in EMAIL_TOOL_DECLARATIONS)
_GITHUB_TOOL_NAMES = frozenset(item["name"] for item in GITHUB_TOOL_DECLARATIONS)
# Email tools that change state and must run at most once per user message.
_EMAIL_WRITE_TOOLS = frozenset(
    {
        "create_email_draft",
        "update_email_draft",
        "prepare_email_to_send",
        "prepare_email_reply",
        "prepare_email_deletion",
        "prepare_email_draft_deletion",
    }
)
# Approval-required tools whose executor needs the token in its own payload.
# Every GitHub write executor validates the token from the payload, so an
# approved call must carry it or the executor re-asks for approval it already has.
_TOKEN_IN_PAYLOAD = frozenset(
    {
        "github_delete_branch",
        "github_merge_pull_request",
    }
)
# Approval-required tools whose executor owns the hold: it stores the proposal and
# issues the token itself. The dispatcher must NOT refuse these before the executor
# runs, or the proposal is never stored and the user's CONFIRM has nothing to
# replay. The executor still refuses to act without a valid token, so nothing runs
# unapproved.
_HOLDING_WRITE_TOOLS = frozenset(
    {
        "github_delete_branch",
        "github_merge_pull_request",
    }
)
_TEXT_ARG_KEYS = ("expression", "text", "json", "code")


def email_read_tool_declarations() -> list[dict[str, Any]]:
    """Email tools that change no data, so a voice session may call them.

    Voice has no approval flow, so only the read-only email tools are
    offered there. The write set lives here, once, and is reused.
    """
    return [
        dict(declaration)
        for declaration in EMAIL_TOOL_DECLARATIONS
        if declaration.get("name") not in _EMAIL_WRITE_TOOLS
    ]

def _gemini_backend_enabled() -> bool:
    return os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "gemini"


def _azure_backend_enabled() -> bool:
    return os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold() == "azure"


# The tools the model is given without asking. Everything else in the catalogue is
# reached through find_tools, so a message that needs nothing pays for four small
# declarations instead of all of them (about 400 tokens rather than 4,863).
#
# Kept on purpose: find_tools (the way to everything else), search_web (the
# instruction tells the model to look up current facts, so it must not need a round
# trip to get it), and the two cheap local tools that plain messages use most.
ALWAYS_ON_TOOL_NAMES = frozenset({"find_tools", "search_web", "calculator", "text_stats"})

# How many tools one find_tools call returns. Five is what the tool is for: enough to
# choose from, few enough that the model still picks rather than guesses.
_FIND_TOOLS_LIMIT = 5


# The read tools a user reaches for first, per connector. These are offered on the
# first round when that connector is connected, so a question about branches does not
# depend on the model choosing to call find_tools first - the failure mode where the
# tool exists, the account is connected, and the model still answers "no access".
#
# Kept deliberately short. All 12 GitHub reads cost about 1,360 tokens and all 11 email
# declarations about 1,516, against ~360 for the whole always-on set; six GitHub reads
# plus the two email reads is about 980, which buys the reliability without giving back
# the saving that lazy discovery exists for.
_CONNECTED_READ_TOOLS: dict[str, tuple[str, ...]] = {
    "github": (
        "github_list_repositories",
        "github_list_branches",
        "github_list_repository_contents",
        "github_read_repository_file",
        "github_list_pull_requests",
        "github_list_repository_commits",
    ),
    "email": ("list_recent_emails", "search_emails"),
    "zoho_mail": ("list_recent_emails", "search_emails"),
}


def _connected_read_tools(user_id: str) -> frozenset[str]:
    """The first-round read tools for the connectors this user has connected.

    A store that cannot answer returns nothing rather than raising: the model then
    falls back to discovery, which is the behaviour that shipped before this.
    """

    from app.ai.execution.account_context import connected_accounts

    try:
        connected = connected_accounts(user_id)
    except Exception:
        return frozenset()
    names: set[str] = set()
    for connector_id in connected:
        names.update(_CONNECTED_READ_TOOLS.get(connector_id, ()))
    return frozenset(names)


def model_tool_declarations(
    user_id: str = "",
    *,
    discovered: Iterable[str] = (),
    full: bool = False,
) -> list[dict[str, Any]]:
    """The tools the model is given for this turn.

    By default: the always-on handful, plus whatever the model discovered with
    ``find_tools`` so far in this exchange. ``full=True`` returns the whole
    catalogue, which voice uses because it filters for the read-only tools and has
    no discovery step.

    Why not send everything: every declaration is re-sent on every round of every
    message. The whole catalogue is 41 declarations, about 4,863 tokens; the
    always-on set is about 400.
    """
    wanted_names = None if full else set(ALWAYS_ON_TOOL_NAMES) | {str(name) for name in discovered}
    if wanted_names is not None and str(user_id or "").strip():
        wanted_names |= _connected_read_tools(str(user_id).strip())

    def wanted(spec: Any) -> bool:
        # A tool the model cannot have the input for is not offered at all: the
        # contact tools need a client-supplied snapshot the model must never invent.
        if not getattr(spec, "model_facing", True):
            return False
        if wanted_names is not None and spec.name not in wanted_names:
            return False
        # A connector the user has not connected is NOT filtered out any more. The
        # executor answers with "connect it in Plugins", which is a usable reply;
        # hiding the tool made the model say it had no access to something the
        # product does. Approval for the write tools is enforced at dispatch.
        return True

    declarations = [tool_declaration(spec) for spec in TOOL_SPECS if wanted(spec)]

    email_names = {str(item["name"]) for item in EMAIL_TOOL_DECLARATIONS}
    if full:
        declarations += [dict(item) for item in EMAIL_TOOL_DECLARATIONS]
    else:
        declarations += [dict(item) for item in EMAIL_TOOL_DECLARATIONS if str(item["name"]) in wanted_names]

    if wanted_names is None or "search_web" in wanted_names:
        declarations.append(dict(SEARCH_WEB_DECLARATION))
    return declarations


def _find_tools_result(query: str, user_id: str = "") -> dict[str, Any]:
    """Run find_tools: search the catalogue and report what can be called next.

    Only tools the model would actually be given are reported, so a discovery never
    promises something the connector filter then refuses.
    """
    return find_tools_payload(query, limit=_FIND_TOOLS_LIMIT)


def _core_payload(spec: Any, arguments: dict[str, Any]) -> str:
    """Turn model arguments into the string payload a core executor expects."""
    if getattr(spec, "payload_style", "json") == "text":
        for key in _TEXT_ARG_KEYS:
            if key in arguments:
                return str(arguments[key])
        for value in arguments.values():
            if isinstance(value, str):
                return value
        return ""
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _first_token(approval_tokens: Iterable[str] | None) -> str:
    for token in approval_tokens or ():
        if isinstance(token, str) and token.strip():
            return token.strip()
    return ""


async def dispatch_tool(
    user_id: str,
    conversation_id: str,
    name: str,
    arguments: dict[str, Any],
    *,
    original_message: str = "",
    approval_tokens: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Run one model-chosen tool and return a JSON-able result for the model."""
    if name in _EMAIL_TOOL_NAMES:
        from app.plugins.email.email_tools import execute_email_tool

        return await execute_email_tool(
            user_id, conversation_id, name, arguments, original_message=original_message
        )

    if name in _GITHUB_TOOL_NAMES:
        from app.plugins.github.chat_tools import _execute_github_tool

        return await _execute_github_tool(user_id, name, arguments)

    if name == "search_web":
        from app.web_research.tools import SearchWebToolCall, execute_search_web_call

        call = SearchWebToolCall(query=str(arguments.get("query") or "").strip(), model_content={})
        result = await execute_search_web_call(call)
        if result.status == "ok":
            return {
                "ok": True,
                "result": {
                    "status": result.status,
                    "results": [
                        {"title": item.title, "url": item.url, "snippet": item.snippet}
                        for item in result.results
                    ],
                },
            }
        message = SEARCH_RESULT_MESSAGES.get(result.status, SEARCH_FAILED)
        return {"ok": False, "code": f"SEARCH_{result.status.upper()}", "message": message}

    spec = get_tool_spec(name)
    if spec is None:
        return {
            "ok": False,
            "code": "UNKNOWN_TOOL",
            "message": UNKNOWN_TOOL,
        }

    # A data-changing tool is a proposal until a server-issued approval arrives.
    # The GitHub writes hold the proposal themselves (and store it, so the user's
    # CONFIRM can replay it), so they are not refused here; every other
    # approval-required tool is refused until a valid token is presented.
    if spec.requires_approval and name not in _HOLDING_WRITE_TOOLS:
        decision = decide_tool(name, approval_tokens=approval_tokens, user_id=user_id)
        if not decision.allowed:
            return {
                "ok": False,
                "code": "APPROVAL_REQUIRED",
                "message": APPROVAL_REQUIRED,
            }

    payload_arguments = dict(arguments)
    # The authenticated identity is injected by the backend, never by the model.
    if user_id.strip():
        payload_arguments.setdefault("user_id", user_id)
        # A GitHub write that needs confirmation is stored against the
        # conversation, so the confirmation can find it again.
        payload_arguments.setdefault("conversation_id", conversation_id)
    # The user's own message decides whether an explicit write request still needs a
    # confirmation. It is the user's wording, never the model's.
    if original_message.strip():
        payload_arguments.setdefault("original_message", original_message)
    if name in _TOKEN_IN_PAYLOAD:
        token = _first_token(approval_tokens)
        if token:
            payload_arguments.setdefault("approval_token", token)

    payload = _core_payload(spec, payload_arguments)
    result = await run_tool_async(name, payload, user_id=user_id)
    if not result.safe:
        return {"ok": False, "code": "TOOL_FAILED", "message": result.output}
    try:
        data: Any = json.loads(result.output)
    except (TypeError, ValueError):
        data = result.output
    # A held write already produced the user-facing proposal and stored itself for
    # the confirmation, so return it as-is rather than burying it under ok:true.
    if isinstance(data, dict) and data.get("code") in {"APPROVAL_REQUIRED", "APPROVAL_UNAVAILABLE"}:
        return data
    return {"ok": True, "result": data}


async def run_tool_loop(
    user_id: str,
    conversation_id: str,
    message: str,
    history: list[tuple[str, str]] | None = None,
    approval_tokens: Iterable[str] | None = None,
    extra_system_instruction: str = "",
) -> str | None:
    """Run the unified tool loop.

    Returns the model's grounded answer when it used a tool, or ``None`` when it
    chose no tool (so the caller can fall back to the plain answer path).
    """
    if not (_gemini_backend_enabled() or _azure_backend_enabled()) or not message.strip():
        return None

    from app.ai.gateway.gemini_service import (
        generate_gemini_function_call_reply,
        run_with_provider_fallback,
    )
    from app.ai.gateway.azure_openai_service import generate_azure_function_call_reply

    tokens = tuple(approval_tokens or ())
    write_done = {"value": False}
    # Tools the model asked for with find_tools, so later rounds can call them.
    discovered: set[str] = set()

    def declarations() -> list[dict[str, Any]]:
        """What the model is offered right now.

        Called once per round by the provider, so a tool discovered in round one is
        callable in round two without re-sending the catalogue.
        """
        return model_tool_declarations(user_id, discovered=tuple(discovered))

    async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "find_tools":
            result = _find_tools_result(str(arguments.get("query") or ""), user_id)
            if result.get("ok") is True:
                discovered.update(
                    str(item["name"]) for item in (result.get("result") or {}).get("tools") or []
                )
            return result
        if name in _EMAIL_WRITE_TOOLS and write_done["value"]:
            return {
                "ok": False,
                "code": "MULTIPLE_WRITES_BLOCKED",
                "message": "One email-changing operation is already pending for this message. Ask the user before starting another.",
            }
        result = await dispatch_tool(
            user_id,
            conversation_id,
            name,
            arguments,
            original_message=message,
            approval_tokens=tokens,
        )
        if name in _EMAIL_WRITE_TOOLS and result.get("ok") is True:
            write_done["value"] = True
        return result

    _shared = dict(
        history=history,
        # A callable, not a list: the provider asks for the declarations each round so
        # a tool found with find_tools becomes callable immediately.
        tool_declarations=declarations,
        execute_tool=execute_tool,
        extra_system_instruction=(
            _UNIFIED_SYSTEM_INSTRUCTION
            + ("\n\n" + extra_system_instruction.strip() if extra_system_instruction.strip() else "")
        ),
        tool_mode="AUTO",
        return_none_when_no_tool=True,
        reuse_first_text_when_no_tool=True,
    )
    _azure_first = _azure_backend_enabled()
    return await run_with_provider_fallback(
        lambda: (generate_azure_function_call_reply if _azure_first else generate_gemini_function_call_reply)(message, **_shared),
        lambda: (generate_gemini_function_call_reply if _azure_first else generate_azure_function_call_reply)(message, **_shared),
    )
