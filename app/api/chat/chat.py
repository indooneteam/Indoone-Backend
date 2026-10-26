import base64
import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.ai.gateway.ai_rules import EMAIL_PENDING_ACTION
from app.ai.memory.conversation_store import ConversationStore
from app.ai.memory.file_context import read_text_file
from app.ai.gateway.service import generate_reply
from app.ai.execution.connector_authorization import authorize_connector
from app.ai.execution import action_chips
from app.plugins.github.service import (
    MAX_REPOSITORY_FILE_BYTES,
    _redact_repository_secrets,
    _validated_repository,
    _validated_repository_path,
)
from app.plugins.github.edit_suggestions import build_code_edit_prompt, parse_code_edit_suggestion
from app.capabilities.media.document_extract import (
    DOCX_MIME,
    MAX_DOCUMENT_BYTES,
    PDF_MIME,
    extract_document,
)
from app.capabilities.media.vision import ALLOWED_IMAGE_MIMES, MAX_IMAGE_BYTES, analyze_image
from app.web_research.grounding import split_grounded_sources
from app.api.security.dependencies import current_user_id
from app.ai.gateway.gemini_service import ModelBusyError
from app.ai.execution.account_context import accounts_instruction, without_stale_refusals
from app.ai.execution.request_context import now_line, user_name_line
from app.ai.execution.confirmation import classify_reply_async
from app.ai.execution import limits as chat_limits
from app.ai.execution.tool_loop import run_tool_loop
from app.ai.agents.agent_async import execute_agent_async
from app.capabilities.connectors.store import (
    get_pending_email_action,
    get_pending_github_action,
    list_memories,
)
from datetime import datetime, timedelta, timezone

_IST = timezone(timedelta(hours=5, minutes=30))
from app.plugins.email.email_tools import (
    email_read_fast_path,
    handle_email_chat,
    handle_email_chat_with_tools,
)
from app.plugins.festivals.calendar import festival_answer
from app.plugins.github.chat_tools import handle_github_chat_with_tools

logger = logging.getLogger("indoone.chat")

router = APIRouter(tags=["chat"])
_store = ConversationStore()
MAX_CHAT_ATTACHMENT_BYTES = max(MAX_DOCUMENT_BYTES, MAX_IMAGE_BYTES)
MAX_CHAT_ATTACHMENT_BASE64_CHARS = ((MAX_CHAT_ATTACHMENT_BYTES + 2) // 3) * 4
MAX_CHAT_TOTAL_ATTACHMENT_BYTES = 20_000_000
MAX_CHAT_ATTACHMENTS = 10
CHAT_ATTACHMENT_MIME_TYPES = ALLOWED_IMAGE_MIMES | {PDF_MIME, DOCX_MIME}


class ChatMediaAttachment(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    mime_type: str = Field(min_length=1, max_length=120)
    content_base64: str = Field(min_length=1, max_length=MAX_CHAT_ATTACHMENT_BASE64_CHARS)


class CodeEditSuggestionRequest(BaseModel):
    repository: str = Field(min_length=3, max_length=200)
    path: str = Field(min_length=1, max_length=1024)
    instruction: str = Field(min_length=1, max_length=2_000)
    content: str = Field(default="", max_length=50_000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)
    file_id: str | None = Field(default=None, min_length=1, max_length=128)
    file_ids: list[str] = Field(default_factory=list, max_length=MAX_CHAT_ATTACHMENTS)
    attachment_filename: str | None = Field(default=None, min_length=1, max_length=255)
    attachment_mime_type: str | None = Field(default=None, min_length=1, max_length=120)
    attachment_base64: str | None = Field(default=None, max_length=MAX_CHAT_ATTACHMENT_BASE64_CHARS)
    attachments: list[ChatMediaAttachment] = Field(default_factory=list, max_length=MAX_CHAT_ATTACHMENTS)
    # Server-issued approval tokens for data-changing tools the user has confirmed.
    approval_tokens: list[str] = Field(default_factory=list, max_length=32)


def _decode_chat_attachment(body: ChatRequest) -> dict[str, object] | None:
    supplied = (body.attachment_filename, body.attachment_mime_type, body.attachment_base64)
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied):
        raise HTTPException(status_code=400, detail="attachment filename, MIME type, and content are all required")
    if body.file_id or body.file_ids:
        raise HTTPException(status_code=400, detail="send file IDs separately from binary attachments")

    filename = Path(str(body.attachment_filename).replace("\\", "/")).name.strip()
    mime_type = str(body.attachment_mime_type).strip().lower()
    if not filename:
        raise HTTPException(status_code=400, detail="invalid attachment filename")
    if mime_type not in CHAT_ATTACHMENT_MIME_TYPES:
        raise HTTPException(status_code=415, detail="supported attachments are JPEG, PNG, WebP, PDF, and DOCX")

    try:
        content = base64.b64decode(str(body.attachment_base64), validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="attachment content is invalid base64") from exc
    if not content:
        raise HTTPException(status_code=400, detail="attachment content is empty")
    if mime_type in ALLOWED_IMAGE_MIMES and len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the 5 MB limit")
    if len(content) > MAX_CHAT_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="attachment exceeds the 8 MB limit")

    return {"filename": filename, "mime_type": mime_type, "content": content}


async def _agent_history_for_chat(
    user_id: str,
    message: str,
    history: list[tuple[str, str]],
    approval_tokens: list[str] | None,
) -> list[tuple[str, str]] | None:
    """Run the multi-step agent and return the history with its tool output appended.

    Returns None when the agent planned nothing, so the caller can fall back to the
    single-tool loop and the plugin paths exactly as before.

    The agent plans its steps deterministically, so it does not need the model
    backend to be enabled, and it can feed one step's result into the next. Its output
    goes into the history as a tool turn, and the model then answers from it.
    """
    execution = await execute_agent_async(
        message,
        user_id=user_id,
        approval_tokens=tuple(approval_tokens or ()),
    )
    if not execution.results:
        return None
    lines = [f"{result.name}: {result.output}" for result in execution.results]
    return [*history, ("tool", "\n".join(lines))]


def _decode_chat_attachments(body: ChatRequest) -> list[dict[str, object]]:
    legacy_fields = (body.attachment_filename, body.attachment_mime_type, body.attachment_base64)
    media_items: list[dict[str, object]] = []

    if body.attachments and any(value is not None for value in legacy_fields):
        raise HTTPException(status_code=400, detail="use either attachments or the legacy single-attachment fields")

    # Check the aggregate on the encoded sizes before decoding anything. Summing after
    # the loop meant every attachment had already been decoded into memory, so the 20 MB
    # budget could not prevent the work it exists to prevent.
    encoded_total = sum(len(item.content_base64) for item in body.attachments)
    if body.attachment_base64:
        encoded_total += len(body.attachment_base64)
    if encoded_total > ((MAX_CHAT_TOTAL_ATTACHMENT_BYTES + 2) // 3) * 4:
        raise HTTPException(status_code=413, detail="Attachments together exceed the 20 MB upload limit.")

    for item in body.attachments:
        decoded = _decode_chat_attachment(
            ChatRequest(
                message=body.message,
                attachment_filename=item.filename,
                attachment_mime_type=item.mime_type,
                attachment_base64=item.content_base64,
            )
        )
        if decoded is not None:
            media_items.append(decoded)

    if any(value is not None for value in legacy_fields):
        decoded = _decode_chat_attachment(body)
        if decoded is not None:
            media_items.append(decoded)

    file_count = len(body.file_ids) + (1 if body.file_id else 0)
    if file_count + len(media_items) > MAX_CHAT_ATTACHMENTS:
        raise HTTPException(status_code=400, detail="You can attach up to 10 files per message.")

    total_bytes = sum(len(bytes(item["content"])) for item in media_items)
    if total_bytes > MAX_CHAT_TOTAL_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail="Attachments together exceed the 20 MB upload limit.")

    if any(not file_id.strip() for file_id in body.file_ids):
        raise HTTPException(status_code=400, detail="invalid file ID")
    return media_items


class ChatSource(BaseModel):
    title: str
    url: str


class ActionChip(BaseModel):
    """One clickable follow-up under a reply."""

    label: str
    # The exact text this chip sends as the user's next message. "CONFIRM" is the same
    # word the reply asks for, so a button and typing cannot diverge.
    action: str


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    sources: list[ChatSource] = Field(default_factory=list)
    # Empty for most replies. A held write carries Confirm/Cancel, so the user presses a
    # button instead of typing a word the reply had to spell out.
    action_chips: list[ActionChip] = Field(default_factory=list)


def _principal(request: Request) -> str:
    return current_user_id(request)


def _client_ip(request: Request) -> str:
    """The caller's address, for the case where there is no authenticated user."""

    client = getattr(request, "client", None)
    return str(getattr(client, "host", "") or "")


def _admit(request: Request, user_id: str) -> None:
    """Refuse a request before it spends anything, or let it through.

    This runs before any attachment is decoded, any history is read and any model is
    called, so a refused request costs a dictionary lookup and one counter write. The
    ``Retry-After`` header is always set on a refusal, so a client can back off without
    guessing.
    """

    admission = chat_limits.admit(user_id, client_ip=_client_ip(request))
    if admission.allowed:
        return
    raise HTTPException(
        status_code=429,
        detail=admission.message or "Too many requests. Try again shortly.",
        headers={"Retry-After": str(max(1, admission.retry_after))},
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    action_chips.begin()
    user_id = _principal(request)
    _admit(request, user_id)
    is_new_conversation = body.conversation_id is None
    conversation_id = body.conversation_id or str(uuid4())

    try:
        if is_new_conversation:
            history: list[tuple[str, str]] = []
        elif _store.is_closed(conversation_id, user_id=user_id):
            conversation_id = str(uuid4())
            history = []
        else:
            # Render Free instances have an ephemeral filesystem, so the backend's
            # local SQLite history can disappear after a restart while Android still
            # has the durable Firestore conversation id. A missing local record must
            # not turn an otherwise valid chat request into HTTP 404.
            try:
                history = _store.recent(conversation_id, user_id=user_id)
            except ValueError:
                history = []
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    document_context_parts: list[str] = []
    media_attachments: list[dict[str, object]] = []
    attachments = _decode_chat_attachments(body)
    file_ids = ([body.file_id] if body.file_id else []) + body.file_ids

    for file_id in file_ids:
        try:
            text = read_text_file(file_id, user_id=user_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if text.strip():
            document_context_parts.append(f"Content from attached text file {file_id}:\n{text[:100_000]}")

    # Media handling needs a provider that accepts images. Both Gemini and Azure
    # do, so this is a provider check, not a gemini-only one.
    _model_backend = os.getenv("INDOONE_MODEL_BACKEND", "").strip().casefold()
    gemini_enabled = _model_backend in {"gemini", "azure"}
    for attachment in attachments:
        filename = str(attachment["filename"])
        mime_type = str(attachment["mime_type"])
        content = bytes(attachment["content"])

        if mime_type in ALLOWED_IMAGE_MIMES:
            if gemini_enabled:
                media_attachments.append(attachment)
            else:
                # Local inference can only use readable text extracted from images.
                try:
                    vision_result = analyze_image(
                        filename,
                        mime_type,
                        base64.b64encode(content).decode("ascii"),
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                ocr_result = vision_result.get("ocr", {})
                ocr_text = str(ocr_result.get("text", "")).strip() if isinstance(ocr_result, dict) else ""
                if not ocr_text:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"The local text model could not read text from {filename}. "
                            "Enable the Gemini backend to understand images."
                        ),
                    )
                document_context_parts.append(
                    f"Readable text extracted from image {filename}:\n{ocr_text[:100_000]}"
                )
        elif mime_type in {PDF_MIME, DOCX_MIME}:
            try:
                document = extract_document(filename, mime_type, content)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if mime_type == PDF_MIME and gemini_enabled:
                # Keep original PDFs so Gemini can inspect scans, charts, and layout.
                media_attachments.append(attachment)
            elif document.text.strip():
                document_context_parts.append(
                    f"Extracted content from {filename}:\n{document.text[:100_000]}"
                )
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{filename} contains no selectable text. "
                        "Use a Gemini-enabled backend for scanned PDFs."
                    ),
                )

    document_context = "\n\n".join(document_context_parts)[:120_000]

    try:
        # A festival or holiday question is answered from the calendar file before
        # anything else runs: no agent, no gateway, no tokens. The dates follow a lunar
        # calendar and a state gazette, so they cannot be computed - and a model asked for
        # one will produce a confident date that is wrong.
        calendar_reply = festival_answer(body.message)

        if calendar_reply is not None:
            # Answered from the file. Nothing below this runs: no agent, no gateway, no
            # tokens - which is the whole point of the module.
            try:
                _store.append(
                    conversation_id,
                    [("user", body.message), ("assistant", calendar_reply)],
                    user_id=user_id,
                )
            except (ValueError, PermissionError):
                pass

            return ChatResponse(
                conversation_id=conversation_id,
                reply=calendar_reply,
                sources=[],
                action_chips=[],
            )

        tool_reply = None
        # Two short lines: the user's name, and today's date and time. Together they
        # cost about 16 tokens; the date line is what stops the model answering a
        # date question from its training memory.
        extra_lines = [
            line
            for line in (user_name_line(user_id), now_line(), accounts_instruction(user_id))
            if line
        ]
        extra_instruction = " ".join(extra_lines)
        # An earlier "I don't have access" must not be replayed now that an account is
        # connected: the model reads its own last answer as context, and an old refusal
        # is a strong pull. Only the assistant's turns are dropped, never the user's.
        history = without_stale_refusals(history, user_id) or []
        if not attachments and not file_ids and not document_context.strip() and not media_attachments:
            # The agent runs first. It plans its own tool steps and can chain one result
            # into the next, and it is deterministic, so it does not depend on the model
            # backend. What it gathers goes into the history and the model answers from
            # it below. When it plans nothing, every path under this runs as it did.
            agent_ran = False
            # The agent decides. Every email tool is either gated on the user having
            # connected that mailbox or gated on approval, so for an email request the
            # agent either plans nothing or blocks the step it planned - and in both
            # cases it produces no result, which leaves the request to the email path
            # below. That path knows the provider the user actually connected and asks
            # for the one confirmation.
            #
            # The pending check is still needed: once an action is waiting, the reply
            # that confirms it may be a bare "yes", which is not a tool request at all.
            # A pending GitHub write lives in the same store, so the email path
            # must not claim it. Only an email integration's pending belongs here.
            _pending_email = get_pending_email_action(user_id, conversation_id)
            email_owns_this = bool(
                _pending_email
                and str(_pending_email.get("integration_id") or "").casefold() != "github"
            )
            # A GitHub write waiting on this conversation is resolved before the
            # agent plans anything: a bare "ok create maadu" names no repository,
            # so nothing else would reach the handler.
            if get_pending_github_action(user_id, conversation_id) is not None:
                tool_reply = await handle_github_chat_with_tools(
                    user_id,
                    body.message,
                    history=history,
                    approval_tokens=tuple(body.approval_tokens or ()),
                    conversation_id=conversation_id,
                )
            # A GitHub request is answered by the deterministic router on EVERY backend,
            # not only the local one. This is the fix for invented branch lists: the model
            # path leaves the choice of calling a tool to the model, and a model that
            # answers from memory produces names that never came from the API - instantly,
            # with no network latency to give it away.
            #
            # The router returns None when the message is not a GitHub request, so this
            # costs a regex pass on everything else.
            # A read-only mailbox question is answered deterministically on every backend.
            # "unread emails" can only mean one thing, so the model was spending tokens to
            # choose a tool the words already named. A write is left alone: it is held for
            # confirmation and that path knows how to hold it.
            if tool_reply is None and not email_owns_this:
                tool_reply = await email_read_fast_path(
                    user_id, conversation_id, body.message
                )

            if tool_reply is None and not email_owns_this:
                tool_reply = await handle_github_chat_with_tools(
                    user_id,
                    body.message,
                    history=history,
                    approval_tokens=tuple(body.approval_tokens or ()),
                    conversation_id=conversation_id,
                )

            # No gate on whether to offer tools: the model is given four small
            # declarations and find_tools, so an ordinary message costs a few hundred
            # tokens rather than the whole catalogue. A message that needs nothing is
            # answered in one call with no tool called, and the agent above still runs
            # first for the short requests a model would have to be told about.
            if tool_reply is None and not email_owns_this:
                agent_history = await _agent_history_for_chat(
                    user_id,
                    body.message,
                    history,
                    body.approval_tokens,
                )
                if agent_history is not None:
                    history = agent_history
                    agent_ran = True
            if gemini_enabled:
                # A pending email confirmation must be resolved before anything else.
                pending = get_pending_email_action(user_id, conversation_id)
                if pending:
                    action_type = str(pending.get("action_type") or "")
                    if (
                        action_type == "await_recipient"
                        or await classify_reply_async(body.message) in {"confirm", "cancel"}
                    ):
                        tool_reply = await handle_email_chat(user_id, conversation_id, body.message)
                    else:
                        tool_reply = (
                            EMAIL_PENDING_ACTION
                        )
                if tool_reply is None and not agent_ran:
                    # One loop, every tool: the model chooses whether and which to call,
                    # and asks for anything else with find_tools. This is the fallback
                    # for when the agent planned nothing.
                    try:
                        tool_reply = await run_tool_loop(
                            user_id,
                            conversation_id,
                            body.message,
                            history=history,
                            approval_tokens=body.approval_tokens,
                            extra_system_instruction=extra_instruction,
                        )
                    except Exception:
                        tool_reply = None
            else:
                # Local backend keeps the deterministic plugin path.
                try:
                    tool_reply = await handle_email_chat_with_tools(
                        user_id,
                        conversation_id,
                        body.message,
                        history=history,
                    )
                except Exception:
                    # Do not conflate model/provider failures with expired mailbox tokens.
                    tool_reply = (
                        "I couldn't complete that email request because an internal AI or "
                        "email service failed. Please retry shortly. I will not claim that "
                        "an email was sent or deleted unless the operation was confirmed."
                    )
                if tool_reply is None and not agent_ran:
                    tool_reply = await handle_github_chat_with_tools(
                        user_id,
                        body.message,
                        history=history,
                        approval_tokens=tuple(body.approval_tokens or ()),
                        conversation_id=conversation_id,
                    )

        if tool_reply is not None:
            reply_with_sources = tool_reply
        elif not media_attachments:
            reply_with_sources = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
                extra_system_instruction=extra_instruction,
            )
        elif len(media_attachments) == 1:
            reply_with_sources = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
                media_attachment=media_attachments[0],
                extra_system_instruction=extra_instruction,
            )
        else:
            reply_with_sources = await generate_reply(
                body.message,
                history=history,
                document_context=document_context,
                media_attachments=media_attachments,
                extra_system_instruction=extra_instruction,
            )
    except ModelBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": "20"},
        ) from exc
    except RuntimeError as exc:
        # The failure is logged in full and the client gets wording of our own. The
        # provider's error text must not travel: it names a third party and points at
        # our configuration rather than at anything the user can act on.
        logger.error("Chat generation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=(
                "Indoone AI could not complete this request. "
                "Please try again in a moment."
            ),
        ) from exc

    reply, web_sources = split_grounded_sources(reply_with_sources)

    try:
        _store.append(
            conversation_id,
            [("user", body.message), ("assistant", reply_with_sources)],
            user_id=user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    chips = action_chips.collected()
    action_chips.reset()

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply,
        sources=[ChatSource(title=source.title, url=source.url) for source in web_sources],
        action_chips=[ActionChip(label=chip["label"], action=chip["action"]) for chip in chips],
    )

@router.post("/code-edit-suggestion")
async def code_edit_suggestion(request: Request, body: CodeEditSuggestionRequest) -> dict[str, object]:
    """Return an AI proposal only; never write or commit the suggested code here."""
    user_id = _principal(request)
    authorization = authorize_connector(user_id, "github", "repo.files.read")
    if not authorization.allowed:
        raise HTTPException(status_code=403, detail=authorization.reason)

    try:
        repository = _validated_repository(body.repository)
        path = _validated_repository_path(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not path:
        raise HTTPException(status_code=400, detail="path must identify a file")
    current_content = body.content
    if len(current_content.encode("utf-8")) > MAX_REPOSITORY_FILE_BYTES:
        raise HTTPException(status_code=413, detail="The source file exceeds the file-edit limit.")
    if current_content != _redact_repository_secrets(current_content):
        raise HTTPException(
            status_code=400,
            detail="The source contains a credential or private key. Redact it before requesting an AI edit.",
        )

    prompt = build_code_edit_prompt(repository, path, body.instruction.strip(), current_content)
    try:
        raw_suggestion = await generate_reply(prompt, history=[], document_context="")
        suggestion = parse_code_edit_suggestion(raw_suggestion)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="AI could not prepare the code proposal. No file was changed.") from exc

    return {
        "repository": repository,
        "path": path,
        "content": suggestion,
        "proposal_only": True,
        "secrets_exposed": False,
    }


