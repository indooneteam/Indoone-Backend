"""Google Gemini API service for the Indoone backend.

The provider is selected explicitly with INDOONE_MODEL_BACKEND=gemini.
The API key stays server-side in GEMINI_API_KEY and is never sent to the Android app.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any
from collections.abc import Awaitable, Callable
from urllib.parse import quote

import httpx

logger = logging.getLogger("indoone.gemini")

# What the user is told when the provider is at its limit. It never names the
# provider, the quota, or the reason: the user only needs to know to retry.
BUSY_MESSAGE = (
    "Indoone is handling a lot of requests right now. Please try again in a moment."
)


class ModelBusyError(RuntimeError):
    """The provider is at its limit right now; the caller should ask again shortly."""

from app.web_research.grounding import (
    GroundedEvidence,
    append_sources,
    build_grounded_prompt_instruction,
)
from app.web_research.tools import (
    build_search_web_function_response,
    build_search_web_tools,
    execute_search_web_call,
    extract_search_web_call,
)
from app.web_research.research import ResearchResult
from app.ai.gateway.observability import (
    add_model_usage,
    measured_model_call,
    note_fallback,
    set_model,
)
from app.ai.gateway.ai_rules import (
    ANSWER_SYSTEM_INSTRUCTION as _SYSTEM_INSTRUCTION,
    EVIDENCE_SUFFIX,
    FUNCTION_DECLARATIONS_REQUIRED,
    INVALID_TOOL_ARGUMENTS,
    INVALID_TOOL_RESULT,
    TOOL_EXECUTION_FAILED,
    TOOL_LIMIT_REACHED,
    UNKNOWN_TOOL,
)


_API_BASE_URL = os.getenv(
    "GEMINI_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")
_DEFAULT_MODEL = "gemma-4-26b-a4b-it"
_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))


def _get_config() -> tuple[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    model = os.getenv("GEMINI_MODEL", _DEFAULT_MODEL).strip()
    if not model:
        raise RuntimeError("GEMINI_MODEL is not configured")

    return api_key, model


def _get_tool_config() -> tuple[str, str]:
    """The model that chooses and drives tools.

    Kept separate from the plain-answer model so the two draw on different
    provider quotas; one message then spreads its usage over both.
    """
    api_key, model = _get_config()
    set_model(model)
    tool_model = os.getenv("GEMINI_TOOL_MODEL", "").strip()
    return api_key, (tool_model or model)


async def run_with_provider_fallback(primary, fallback):
    """Run the primary provider; when it is at its limit, run the same call on the other.

    One bridge, two providers. The tools, the tool loop, the retry and the logging
    all stay where they are - only the API call underneath changes. The primary is
    Gemini unless INDOONE_MODEL_BACKEND forces Azure; when the primary reports that
    it is at its limit (ModelBusyError), the same call is made on the other provider,
    so the user gets an answer instead of an error.

    The Azure service is imported inside the function: it imports this module for
    the shared system instruction and error types, so a top-level import here would
    be circular.
    """
    try:
        return await primary()
    except ModelBusyError:
        logger.warning("The primary provider is at its limit; falling back to the other provider")
        note_fallback()
        return await fallback()


def _content_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def _build_contents(
    message: str,
    history: list[tuple[str, str]] | None,
    document_context: str,
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []

    if history:
        for role, content in history[-12:]:
            clean_content = _content_text(content)
            if not clean_content or role not in {"user", "assistant"}:
                continue
            contents.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": clean_content[:4_000]}],
                }
            )

    user_content = message.strip()
    context = document_context.strip()
    if context:
        if context.startswith("LIVE WEB EVIDENCE:\n"):
            context = context[len("LIVE WEB EVIDENCE:\n"):]
            user_content += (
                "\n\nVERIFIED WEB EVIDENCE:\n"
                + context[:7_000]
                + EVIDENCE_SUFFIX
            )
        elif context.startswith("LIVE WEB RESEARCH EVIDENCE:\n"):
            context = context[len("LIVE WEB RESEARCH EVIDENCE:\n"):]
            user_content += (
                "\n\nVERIFIED WEB EVIDENCE:\n"
                + context[:7_000]
                + EVIDENCE_SUFFIX
            )
        else:
            user_content += "\n\nUSER-PROVIDED DOCUMENT:\n" + context[:12_000]

    user_parts: list[dict[str, Any]] = [{"text": user_content}]
    attachments = list(media_attachments or [])
    if media_attachment is not None:
        attachments.insert(0, media_attachment)
    for attachment in attachments:
        mime_type = str(attachment.get("mime_type", "")).strip().lower()
        raw_content = attachment.get("content")
        if not mime_type or not isinstance(raw_content, (bytes, bytearray)):
            raise ValueError("invalid multimodal attachment")
        user_parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(bytes(raw_content)).decode("ascii"),
                }
            }
        )

    contents.append({"role": "user", "parts": user_parts})
    return contents


def _extract_text(response_data: dict[str, Any]) -> str:
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("Gemini returned no candidates")

    candidate = candidates[0]
    if not isinstance(candidate, dict):
        raise RuntimeError("Gemini returned an invalid candidate")

    content = candidate.get("content")
    if not isinstance(content, dict):
        raise RuntimeError("Gemini returned no response content")

    parts = content.get("parts")
    if not isinstance(parts, list):
        raise RuntimeError("Gemini returned no response parts")

    text_parts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        # Never expose any model part explicitly marked as internal thought.
        if part.get("thought") is True:
            continue
        if isinstance(part.get("text"), str):
            text_parts.append(part["text"])

    reply = "".join(text_parts).strip()
    if not reply:
        raise RuntimeError("Gemini returned an empty user-facing answer")

    return reply


def _generation_config(model: str) -> dict[str, Any]:
    # Do not impose an app-level output-token cap. When maxOutputTokens is
    # omitted, the Gemini API uses the selected model's own output-token limit.
    config: dict[str, Any] = {
        "temperature": 0.2,
        "topP": 0.9,
    }
    # Gemma 4 exposes a thinking-level control through the GenerateContent API.
    # Minimal disables its thinking mode, preventing internal deliberation from
    # being generated as ordinary response text. Do not send this setting to
    # other model families, which can have different thinking configurations.
    if model.casefold().startswith("gemma-4-"):
        config["thinkingConfig"] = {"thinkingLevel": "minimal"}
    return config


# The provider intermittently answers 500 "Internal error" or a gateway error
# (measured: roughly one call in ten). One short retry clears it, so a provider
# hiccup does not become a lost reply for the user.
_TRANSIENT_STATUSES = frozenset({500, 502, 503, 504})
_MAX_ATTEMPTS = 3
_RETRY_DELAYS = (1.0, 2.0)


async def _post_generate_content(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    response: httpx.Response | None = None
    for attempt in range(_MAX_ATTEMPTS):
        last_attempt = attempt + 1 >= _MAX_ATTEMPTS
        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            if last_attempt:
                raise RuntimeError("Gemini request failed") from exc
            await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])
            continue
        if response.status_code not in _TRANSIENT_STATUSES or last_attempt:
            break
        logger.warning(
            "Gemini returned HTTP %s; retrying (%d/%d)",
            response.status_code,
            attempt + 1,
            _MAX_ATTEMPTS,
        )
        await asyncio.sleep(_RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)])

    if response.status_code != 200:
        detail = response.text[:500].replace("\n", " ").strip()
        if response.status_code == 429:
            # The provider names the quota that ran out (requests or tokens,
            # per minute or per day). Log it so the real limit is known.
            logger.warning("Gemini rate limit reached: %s", detail[:400])
            raise ModelBusyError(BUSY_MESSAGE)
        # The provider's own text is logged here and never placed in the exception.
        # This message travels to the client, so it must not carry a third party's
        # error body. The status code is ours and is safe to keep.
        logger.error(
            "Gemini returned HTTP %s: %s",
            response.status_code,
            detail,
        )
        if response.status_code in {401, 403}:
            raise RuntimeError("Gemini API authentication failed")
        raise RuntimeError(f"Gemini API returned HTTP {response.status_code}")

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError("Gemini API returned invalid JSON") from exc
    if not isinstance(response_data, dict):
        raise RuntimeError("Gemini API returned invalid response data")
    _record_usage(response_data)
    return response_data


def _delta_text(frame: dict[str, Any]) -> str:
    """The text in one streamed Gemini frame, or "" when the frame carries none.

    A frame can carry usage metadata, a finish reason, or nothing at all, so every level
    is checked rather than assumed.
    """
    candidates = frame.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        return ""

    first = candidates[0]
    content = first.get("content") if isinstance(first, dict) else None
    parts = content.get("parts") if isinstance(content, dict) else None

    if not isinstance(parts, list):
        return ""

    return "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict)
    )


def _record_usage(response_data: dict[str, Any]) -> None:
    """Add the token counts Gemini reported to the open model span.

    Tokens are what a cost is computed from, and this is the only place the
    provider reports them. A payload without usage is left alone rather than
    guessed at.
    """

    usage = response_data.get("usageMetadata")
    if not isinstance(usage, dict):
        return
    try:
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
    except (TypeError, ValueError):
        return
    add_model_usage(input_tokens, output_tokens)


@measured_model_call("gemini")
async def generate_gemini_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
    extra_system_instruction: str = "",
) -> str:
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")

    api_key, model = _get_config()
    set_model(model)
    encoded_model = quote(model, safe="")
    url = f"{_API_BASE_URL}/models/{encoded_model}:generateContent"
    system_instruction = _SYSTEM_INSTRUCTION + "\n\n" + build_grounded_prompt_instruction()
    if extra_system_instruction.strip():
        system_instruction += "\n\n" + extra_system_instruction.strip()

    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": _build_contents(cleaned, history, document_context, media_attachment, media_attachments),
        "generationConfig": _generation_config(model),
    }
    tools = build_search_web_tools(model)
    if tools:
        payload["tools"] = tools

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(_TIMEOUT_SECONDS, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response_data = await _post_generate_content(client, url, headers, payload)
        search_call = extract_search_web_call(response_data)
        evidence: list[GroundedEvidence] = []
        if search_call is not None:
            tool_result = await execute_search_web_call(search_call)
            evidence = [
                GroundedEvidence(item.title, item.url, item.snippet)
                for item in tool_result.results
            ]
            model_content = dict(search_call.model_content)
            model_content["role"] = "model"
            followup_payload = dict(payload)
            followup_payload["contents"] = [
                *payload["contents"],
                model_content,
                build_search_web_function_response(search_call, tool_result),
            ]
            # Limit this request to one bounded search operation. The final
            # model turn synthesizes the returned evidence rather than calling
            # another tool indefinitely.
            followup_payload.pop("tools", None)
            response_data = await _post_generate_content(client, url, headers, followup_payload)

    reply = _extract_text(response_data)
    return append_sources(reply, evidence) if evidence else reply


def _extract_function_calls(response_data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the model content and its requested function calls, if any."""
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return {}, []
    content = candidates[0].get("content")
    if not isinstance(content, dict):
        return {}, []
    calls: list[dict[str, Any]] = []
    parts = content.get("parts")
    if not isinstance(parts, list):
        return dict(content), calls
    for part in parts:
        if not isinstance(part, dict) or part.get("thought") is True:
            continue
        call = part.get("functionCall") or part.get("function_call")
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        arguments = call.get("args", {})
        if name:
            calls.append({
                "name": name,
                "args": arguments if isinstance(arguments, dict) else {},
                "id": str(call.get("id") or "").strip(),
                "invalid_arguments": not isinstance(arguments, dict),
            })
    return dict(content), calls


@measured_model_call("gemini")
async def generate_gemini_function_call_reply(
    message: str,
    *,
    history: list[tuple[str, str]] | None,
    tool_declarations: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]],
    execute_tool: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    extra_system_instruction: str,
    max_tool_rounds: int = 4,
    tool_mode: str = "ANY",
    return_none_when_no_tool: bool = False,
    reuse_first_text_when_no_tool: bool = False,
) -> str | None:
    """Run a bounded Gemini function-calling loop and return its grounded final text.

    Function calls are proposals only. The authenticated backend callback owns
    allow-list enforcement, account scoping, parameter validation and confirmations.
    """
    cleaned = message.strip()
    if not cleaned:
        raise ValueError("message cannot be empty")
    if not tool_declarations:
        raise ValueError(FUNCTION_DECLARATIONS_REQUIRED)

    api_key, model = _get_tool_config()
    set_model(model)
    encoded_model = quote(model, safe="")
    url = f"{_API_BASE_URL}/models/{encoded_model}:generateContent"
    system_instruction = (
        _SYSTEM_INSTRUCTION
        + "\n\n"
        + build_grounded_prompt_instruction()
        + "\n\n"
        + extra_system_instruction.strip()
    )
    contents = _build_contents(cleaned, history, "")
    def declarations() -> list[dict[str, Any]]:
        """The declarations to offer, asked for once per round.

        A callable lets the caller grow the set while the loop runs - a tool the
        model discovered with find_tools becomes callable on the next round, without
        re-sending the whole catalogue.
        """
        source = tool_declarations() if callable(tool_declarations) else tool_declarations
        valid = [item for item in source if isinstance(item, dict) and item.get("name")]
        if not valid:
            raise ValueError("no valid function declarations were supplied")
        return valid

    declarations()
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(_TIMEOUT_SECONDS, connect=15.0)
    max_rounds = max(1, min(int(max_tool_rounds), 6))
    max_total_calls = 8
    executed_calls = 0
    tool_names: set[str] = set()

    async with httpx.AsyncClient(timeout=timeout) as client:
        for round_index in range(max_rounds + 1):
            payload: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_instruction}]},
                "contents": contents,
                "generationConfig": _generation_config(model),
            }
            # The final turn has no tools, stopping recursive tool execution.
            if round_index < max_rounds:
                offered = declarations()
                tool_names = {str(item["name"]) for item in offered}
                payload["tools"] = [{"functionDeclarations": offered}]
                if round_index == 0:
                    # "ANY" forces a tool call on the first turn (used when the
                    # request must be grounded in live data). "AUTO" lets the
                    # model decide whether any tool is needed at all.
                    payload["toolConfig"] = {
                        "functionCallingConfig": {"mode": tool_mode}
                    }
            response_data = await _post_generate_content(client, url, headers, payload)
            model_content, calls = _extract_function_calls(response_data)
            if not calls:
                if executed_calls == 0:
                    if return_none_when_no_tool:
                        if not reuse_first_text_when_no_tool:
                            # The caller decides what to do when the model chose
                            # no tool (for example, fall back to a plain answer).
                            return None
                        # The model answered without a tool; that text is the
                        # answer, so reuse it rather than asking again.
                        try:
                            text = _extract_text(response_data)
                        except RuntimeError:
                            return None
                        return text or None
                    # The first email response must perform a real tool call;
                    # never treat a plain model answer as evidence of mailbox data.
                    return (
                        "I couldn't verify the mailbox data because no email tool "
                        "was executed. Please retry; I won't guess about your emails."
                    )
                return _extract_text(response_data)
            if round_index >= max_rounds:
                return (
                    "I couldn't complete that email request safely because the tool "
                    "sequence did not finish. No unconfirmed send or delete was performed."
                )

            model_content["role"] = "model"
            model_parts = model_content.get("parts")
            if isinstance(model_parts, list):
                model_content["parts"] = [
                    part for part in model_parts
                    if not (isinstance(part, dict) and part.get("thought") is True)
                ]
            contents.append(model_content)
            response_parts: list[dict[str, Any]] = []
            for call in calls:
                name = str(call["name"])
                call_id = str(call.get("id") or "")
                if executed_calls >= max_total_calls:
                    result: dict[str, Any] = {
                        "ok": False,
                        "code": "TOOL_LIMIT_REACHED",
                        "message": TOOL_LIMIT_REACHED,
                    }
                elif name not in tool_names:
                    result = {
                        "ok": False,
                        "code": "UNKNOWN_TOOL",
                        "message": UNKNOWN_TOOL,
                    }
                elif call.get("invalid_arguments"):
                    result = {
                        "ok": False,
                        "code": "INVALID_TOOL_ARGUMENTS",
                        "message": INVALID_TOOL_ARGUMENTS,
                    }
                else:
                    try:
                        result = await execute_tool(name, call["args"])
                        if not isinstance(result, dict):
                            result = {
                                "ok": False,
                                "code": "INVALID_TOOL_RESULT",
                                "message": INVALID_TOOL_RESULT,
                            }
                    except Exception:
                        # Exception contents can contain provider response data. Keep
                        # them out of the model context and user response.
                        result = {
                            "ok": False,
                            "code": "TOOL_EXECUTION_FAILED",
                            "message": TOOL_EXECUTION_FAILED,
                        }
                executed_calls += 1

                function_response: dict[str, Any] = {
                    "name": name,
                    "response": {"result": result},
                }
                if call_id:
                    function_response["id"] = call_id
                response_parts.append({"functionResponse": function_response})

            contents.append({"role": "user", "parts": response_parts})

    return (
        "I couldn't complete that email request safely. Please try again; "
        "no action without explicit confirmation was performed."
    )


async def stream_gemini_reply(
    message: str,
    history: list[tuple[str, str]] | None = None,
    document_context: str = "",
    extra_system_instruction: str = "",
    media_attachment: dict[str, object] | None = None,
    media_attachments: list[dict[str, object]] | None = None,
):
    """Yield Gemini's own token deltas as they arrive.

    Same request as the blocking call, with :streamGenerateContent?alt=sse. Usage metadata
    arrives on the frames too, so quota accounting still sees the real token counts rather
    than a guess.
    """
    cleaned = message.strip()

    if not cleaned:
        raise ValueError("message cannot be empty")

    api_key, model = _get_config()
    set_model(model)
    encoded_model = quote(model, safe="")
    url = f"{_API_BASE_URL}/models/{encoded_model}:streamGenerateContent?alt=sse"

    system_instruction = _SYSTEM_INSTRUCTION + "\n\n" + build_grounded_prompt_instruction()

    if extra_system_instruction.strip():
        system_instruction += "\n\n" + extra_system_instruction.strip()

    payload: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": _build_contents(
            cleaned, history, document_context, media_attachment, media_attachments
        ),
        "generationConfig": _generation_config(model),
    }

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    timeout = httpx.Timeout(_TIMEOUT_SECONDS, connect=15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", "ignore")[:400]
                detail = detail.replace("\n", " ").strip()

                if response.status_code == 429:
                    logger.warning("Gemini stream rate limit reached: %s", detail[:300])
                    raise ModelBusyError(BUSY_MESSAGE)

                logger.error(
                    "Gemini stream returned HTTP %s: %s", response.status_code, detail
                )

                if response.status_code in {401, 403}:
                    raise RuntimeError("Gemini API authentication failed")

                raise RuntimeError(f"Gemini API returned HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                raw = line[len("data: "):].strip()

                if not raw or raw == "[DONE]":
                    continue

                try:
                    frame = json.loads(raw)
                except ValueError:
                    continue

                if not isinstance(frame, dict):
                    continue

                # The provider reports tokens on the frames; record them as they come.
                _record_usage(frame)

                text = _delta_text(frame)

                if text:
                    yield text
