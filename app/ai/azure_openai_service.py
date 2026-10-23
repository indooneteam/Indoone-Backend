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
        function["parameters"] = parameters if isinstance(parameters, dict) else {
            "type": "object",
            "properties": {},
        }
        tools.append({"type": "function", "function": function})
    return tools


async def _post_chat(
    url: str,
    api_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    headers = {
        "api-key": api_key,
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code in _BUSY_STATUS:
        raise ModelBusyError(BUSY_MESSAGE)
    if response.status_code >= 400:
        detail = ""
        try:
            detail = json.dumps(response.json())[:400]
        except Exception:  # noqa: BLE001 - the body may not be JSON at all
            detail = response.text[:400]
        raise RuntimeError(
            f"Azure OpenAI returned {response.status_code}: {detail}"
        )

    return response.json()


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

    for round_index in range(max(1, max_tool_rounds)):
        payload: dict[str, Any] = {"messages": messages, "tools": tools}
        if tool_mode.upper() == "ANY" and round_index == 0:
            # "required" is OpenAI's spelling of Gemini's ANY: at least one call.
            payload["tool_choice"] = "required"

        response_data = await _post_chat(url, api_key, payload)
        assistant_message = _first_message(response_data)
        tool_calls = assistant_message.get("tool_calls") or []

        if not tool_calls:
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

            result = await execute_tool(name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, default=str)[:20_000],
                }
            )

        # A short pause between rounds, as the Gemini loop does, so a burst of
        # rounds does not hit the deployment's per-second limit.
        await asyncio.sleep(0)

    return first_text
