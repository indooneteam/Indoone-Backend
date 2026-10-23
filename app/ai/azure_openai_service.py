"""Azure OpenAI chat service for the Indoone backend.

The provider is selected explicitly with INDOONE_MODEL_BACKEND=azure, exactly as
Google Gemini is selected with =gemini. Nothing here changes the Gemini path:
this module is additive, and the switch back is one environment variable.

Three things differ from a plain OpenAI call, and they are the whole point of
this file:

  * the URL is the deployment URL -
    {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...
  * the key travels in the "api-key" header, not "Authorization: Bearer"
  * the model is named by the deployment you created in Azure, not by a model id

The API key stays server-side in AZURE_OPENAI_API_KEY and is never sent to the
Android app or the web.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from collections.abc import Awaitable, Callable

import httpx

# One source of truth for the assistant's own words and for the error the rest of
# the code catches. Both live with the Gemini service today; reusing them here
# means the two providers cannot drift apart, and the Gemini file is untouched.
from app.ai.gemini_service import BUSY_MESSAGE, ModelBusyError, _SYSTEM_INSTRUCTION
from app.web_research.grounding import build_grounded_prompt_instruction

logger = logging.getLogger("indoone.azure_openai")

__all__ = [
    "BUSY_MESSAGE",
    "ModelBusyError",
    "generate_azure_reply",
    "generate_azure_function_call_reply",
    "azure_configuration_ready",
    "azure_model_name",
]

_DEFAULT_API_VERSION = "2024-10-21"
_TIMEOUT_SECONDS = float(os.getenv("AZURE_OPENAI_TIMEOUT_SECONDS", "60"))

# The same shape the Gemini transport uses: a few attempts, backing off, and the
# provider's own message written to the log so a failure can be read later.
_MAX_ATTEMPTS = 3
_RETRY_DELAYS = (1.0, 2.0, 4.0)
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

# Azure answers a spent quota or a throttled deployment with these. The caller
# turns them into ModelBusyError so the user is asked to try again shortly.
_BUSY_STATUS = {429, 503}


def _get_config() -> tuple[str, str, str, str]:
    """Return (key, endpoint, deployment, api_version), all from the environment."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not configured")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not configured")

    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not configured")

    api_version = (
        os.getenv("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION).strip()
        or _DEFAULT_API_VERSION
    )

    return api_key, endpoint, deployment, api_version


def _get_tool_config() -> tuple[str, str, str, str]:
    """The deployment that drives tools, which may be a second deployment.

    Kept separate from the plain-answer deployment for the same reason Gemini
    keeps GEMINI_TOOL_MODEL separate: one message then spreads its usage over
    two quotas instead of one.
    """
    api_key, endpoint, deployment, api_version = _get_config()
    tool_deployment = os.getenv("AZURE_OPENAI_TOOL_DEPLOYMENT", "").strip()
    return api_key, endpoint, (tool_deployment or deployment), api_version


def azure_model_name() -> str | None:
    """The configured deployment name, safe to report. Never the key."""
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    return deployment or None


def azure_configuration_ready() -> bool:
    """True when a key, an endpoint and a deployment are all present."""
    return bool(
        os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        and os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        and os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    )


def _chat_url(endpoint: str, deployment: str, api_version: str) -> str:
    return (
        f"{endpoint}/openai/deployments/{deployment}"
        f"/chat/completions?api-version={api_version}"
    )


def _content_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def _build_messages(
    message: str,
    history: list[tuple[str, str]] | None,
    document_context: str,
    system_instruction: str,
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
) -> list[dict[str, Any]]:
    """Build the OpenAI-style messages array from the same inputs Gemini gets."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_instruction}
    ]

    if history:
        for role, content in history[-12:]:
            clean_content = _content_text(content)
            if not clean_content or role not in {"user", "assistant"}:
                continue
            messages.append(
                {
                    "role": "user" if role == "user" else "assistant",
                    "content": clean_content[:4_000],
                }
            )

    user_content = message.strip()
    context = document_context.strip()
    if context:
        for prefix in ("LIVE WEB EVIDENCE:\n", "LIVE WEB RESEARCH EVIDENCE:\n"):
            if context.startswith(prefix):
                context = context[len(prefix):]
                break
        user_content += (
            "\n\nVERIFIED WEB EVIDENCE:\n"
            + context[:7_000]
            + "\n\nUse this evidence only to answer the user's question. "
            "Do not reproduce the evidence block or source formatting."
        )

    parts: list[dict[str, Any]] = [{"type": "text", "text": user_content}]

    attachments: list[dict[str, object]] = []
    if media_attachments:
        attachments.extend(media_attachments)
    elif media_attachment is not None:
        attachments.append(media_attachment)

    for attachment in attachments:
        encoded = attachment.get("content_base64") or attachment.get("base64") or ""
        mime_type = str(attachment.get("mime_type") or "image/jpeg")
        if not encoded:
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )

    if len(parts) == 1:
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": parts})

    return messages



# Gemini writes its schema types UPPERCASE - "STRING", "OBJECT", "NUMBER" - and
# OpenAI rejects those with a 400. The app's tool declarations are written for
# Gemini, so every tool-carrying request failed and the turn fell back to a plain
# answer with no tools, which is why the assistant kept saying it had no access.
# This walks the schema and lowercases the type names.
_SCHEMA_TYPES = {
    "STRING": "string",
    "NUMBER": "number",
    "INTEGER": "integer",
    "BOOLEAN": "boolean",
    "OBJECT": "object",
    "ARRAY": "array",
    "NULL": "null",
}


def lowercase_schema(value: Any) -> Any:
    """Return the same schema with every type name lowercased, recursively."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == "type" and isinstance(item, str) and item.upper() in _SCHEMA_TYPES:
                out[key] = _SCHEMA_TYPES[item.upper()]
            else:
                out[key] = lowercase_schema(item)
        return out
    if isinstance(value, list):
        return [lowercase_schema(item) for item in value]
    return value


def _openai_tools(tool_declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the declarations the caller already builds into OpenAI's shape.

    The rest of the codebase builds {name, description, parameters} for Gemini;
    OpenAI wants the same thing wrapped in {"type": "function", "function": ...}.
    """
    tools: list[dict[str, Any]] = []
    for declaration in tool_declarations or []:
        name = str(declaration.get("name") or "").strip()
        if not name:
            continue
        function: dict[str, Any] = {"name": name}
        if declaration.get("description"):
            function["description"] = declaration["description"]
        parameters = declaration.get("parameters")
        function["parameters"] = (
            lowercase_schema(parameters)
            if isinstance(parameters, dict)
            else {"type": "object", "properties": {}}
        )
        tools.append({"type": "function", "function": function})
    return tools


async def _post_chat(
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST to Azure, retrying transient failures, logging the provider's own words.

    This mirrors the Gemini transport: the provider's error text is written to the
    log and never put into the exception, because that message travels to the
    client and must not carry a third party's error body.
    """
    headers = {
        "api-key": api_key,
        "content-type": "application/json",
    }
    response = None
    for attempt in range(_MAX_ATTEMPTS):
        last_attempt = attempt + 1 >= _MAX_ATTEMPTS
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            if last_attempt:
                logger.error("Azure OpenAI request failed: %s", exc)
                raise RuntimeError("Azure OpenAI request failed") from exc
            await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            continue

        if response.status_code not in _TRANSIENT_STATUSES or last_attempt:
            break

        logger.warning(
            "Azure OpenAI returned HTTP %s; retrying (%d/%d)",
            response.status_code,
            attempt + 1,
            _MAX_ATTEMPTS,
        )
        await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

    if response is None:
        raise RuntimeError("Azure OpenAI request failed")

    if response.status_code != 200:
        detail = response.text[:500].replace("\n", " ").strip()
        if response.status_code == 429:
            logger.warning("Azure OpenAI rate limit reached: %s", detail[:400])
            raise ModelBusyError(BUSY_MESSAGE)
        logger.error("Azure OpenAI returned HTTP %s: %s", response.status_code, detail)
        if response.status_code in {401, 403}:
            raise RuntimeError("Azure OpenAI authentication failed")
        raise RuntimeError(f"Azure OpenAI returned HTTP {response.status_code}")

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError("Azure OpenAI returned invalid JSON") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError("Azure OpenAI returned invalid response data")
    return response_data


def _first_message(response_data: dict[str, Any]) -> dict[str, Any]:
    choices = response_data.get("choices") or []
    if not choices:
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _extract_text(response_data: dict[str, Any]) -> str:
    content = _first_message(response_data).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Some deployments answer with content parts rather than a string.
        pieces = [
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(piece for piece in pieces if piece).strip()
    return ""


def _system_instruction(extra_system_instruction: str) -> str:
    instruction = _SYSTEM_INSTRUCTION + "\n\n" + build_grounded_prompt_instruction()
    if extra_system_instruction.strip():
        instruction += "\n\n" + extra_system_instruction.strip()
    return instruction


async def generate_azure_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
    extra_system_instruction: str = "",
) -> str:
    """A plain Azure OpenAI reply, with the same signature as the Gemini one."""
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    api_key, endpoint, deployment, api_version = _get_config()
    url = _chat_url(endpoint, deployment, api_version)

    payload: dict[str, Any] = {
        "messages": _build_messages(
            cleaned,
            history,
            document_context,
            _system_instruction(extra_system_instruction),
            media_attachment,
            media_attachments,
        ),
    }

    response_data = await _post_chat(url, api_key, payload)
    reply = _extract_text(response_data)
    if not reply:
        raise RuntimeError("Azure OpenAI returned an empty reply")
    return reply


async def generate_azure_function_call_reply(
    message: str,
    *,
    history: list[tuple[str, str]] | None,
    tool_declarations: list[dict[str, Any]],
    execute_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    extra_system_instruction: str,
    max_tool_rounds: int = 4,
    tool_mode: str = "ANY",
    return_none_when_no_tool: bool = False,
    reuse_first_text_when_no_tool: bool = False,
) -> str | None:
    """Run a bounded Azure OpenAI function-calling loop.

    The same contract as the Gemini loop: the model only *proposes* a call, and
    the authenticated backend callback owns allow-list enforcement, account
    scoping, parameter validation and confirmations. The loop is bounded by
    max_tool_rounds so a model that keeps calling cannot spin.
    """
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")
    if not tool_declarations:
        raise ValueError("at least one function declaration is required")

    api_key, endpoint, deployment, api_version = _get_tool_config()
    url = _chat_url(endpoint, deployment, api_version)

    messages = _build_messages(
        cleaned,
        history,
        "",
        _system_instruction(extra_system_instruction),
    )
    tools = _openai_tools(tool_declarations)

    first_text: str | None = None

    # The same bounds the Gemini loop uses: one extra round so the model can give
    # a final answer after its tool calls, and a hard cap on total tool calls so a
    # model that keeps calling cannot spin.
    max_rounds = max(1, min(int(max_tool_rounds), 6))
    max_total_calls = 8
    executed_calls = 0

    logger.info(
        "Azure OpenAI tool round: %d tool(s) offered: %s",
        len(tools),
        ", ".join(t["function"]["name"] for t in tools) or "(none)",
    )

    for round_index in range(max_rounds + 1):
        payload: dict[str, Any] = {"messages": messages}
        # The final round carries no tools, so the model must answer in words.
        if round_index < max_rounds:
            payload["tools"] = tools
            if tool_mode.upper() == "ANY" and round_index == 0:
                # "required" is OpenAI's spelling of Gemini's ANY: at least one call.
                payload["tool_choice"] = "required"

        response_data = await _post_chat(url, api_key, payload)
        assistant_message = _first_message(response_data)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
            logger.info("Azure OpenAI round %d: no tool called", round_index)
            text = _extract_text(response_data)
            if first_text is None:
                first_text = text or None
            if return_none_when_no_tool:
                return None
            if reuse_first_text_when_no_tool and first_text:
                return first_text
            return text or first_text

        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
            }
        )

        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "").strip()
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            executed_calls += 1
            logger.info("Azure OpenAI called tool: %s", name)
            result = await execute_tool(name, arguments)
            # Log the outcome, never the payload: email and repository content is
            # private. This is the line that says whether a tool worked or refused,
            # which is the difference between "no tools offered" and "the tool
            # itself answered with an error".
            if isinstance(result, dict):
                logger.info(
                    "Azure OpenAI tool %s returned ok=%s code=%s error=%s keys=%s",
                    name,
                    result.get("ok"),
                    result.get("code"),
                    str(result.get("error"))[:200],
                    ",".join(sorted(str(k) for k in result.keys()))[:300],
                )
            else:
                logger.info("Azure OpenAI tool %s returned %s", name, type(result).__name__)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, default=str)[:20_000],
                }
            )

        if executed_calls >= max_total_calls:
            logger.warning("Azure OpenAI tool call cap reached (%d)", max_total_calls)
            break

        # A short pause between rounds, as the Gemini loop does, so a burst of
        # rounds does not hit the deployment's per-second limit.
        await asyncio.sleep(0)

    return first_text
