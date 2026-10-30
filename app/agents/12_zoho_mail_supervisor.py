"""Agent 12 - the mail supervisor: reading is free, sending needs a person.

GitHub holds code. A mailbox holds OTPs, bank alerts, private conversations, and the one
thing that cannot be undone: a message already sent. So this agent draws a line the model
cannot cross on its own.

**Reading is automatic.** Search, read a thread, prepare a draft - all of it happens without
asking, because none of it changes anything in the world.

**Sending needs a person.** A draft produces a one-time token. The token is returned to the
*client*, which renders it as a chip the user presses. It is deliberately **not** part of the
text the model reads: a model that can see the token can send without anyone agreeing, which
is precisely the failure this exists to prevent. A send without a matching, unused,
unexpired token is refused with UNAUTHORIZED_MUTATION and never reaches Zoho.

Two more things a mailbox needs that a repository does not:

* **An OTP is a credential that expires in ninety seconds.** It has no business in a prompt.
  Inbound bodies are scrubbed - OTPs, passwords, private keys, tracking pixels, inline
  base64 images - and a thread past 8,000 characters is truncated with a notice, because a
  long thread will otherwise eat the context the answer needs.
* **A mailbox fails differently.** An expired token, a rate limit, a thread that has been
  deleted: each becomes something the model can say in the user's language.

It executes nothing itself. The Zoho client, the pending-action store and the secret
patterns already exist; this is the policy that decides what may be called and what may be
handed back.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_BODY_CHARS = 8_000
MAX_SEARCH_RESULTS = 10
TOKEN_TTL_SECONDS = 15 * 60
MAX_PENDING_DRAFTS = 512

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# A one-time code: four to eight digits, usually announced as one.
_OTP = re.compile(
    r"(?i)\b(?:otp|o\.t\.p|one[\s-]?time\s+(?:code|password)|verification\s+code|"
    r"security\s+code|confirmation\s+code)\b[^0-9]{0,20}(\d{4,8})\b"
)
_BARE_OTP = re.compile(r"(?i)\b(?:your|is|code)\s*[:\-]?\s*(\d{4,8})\b(?=[^\d]|$)")
_PASSWORD_LINE = re.compile(
    r"(?im)^\s*[+-]?\s*(?:password|passwd|pwd|passcode)\s*[:=]\s*\S+"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
)
_BASE64_IMAGE = re.compile(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]{40,}")
_TRACKING_PIXEL = re.compile(r"<img[^>]*(?:width|height)\s*=\s*[\"']?1[\"']?[^>]*>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")

REDACTED_OTP = "[OTP Redacted]"
REDACTED = "[REDACTED_SECRET]"

_MESSAGES = {
    "UNAUTHORIZED_MUTATION": {
        "kn": "ಇಮೇಲ್ ಕಳಿಸಲು ನಿಮ್ಮ ದೃಢೀಕರಣ ಬೇಕು. ಡ್ರಾಫ್ಟ್ ತಯಾರಿದೆ.",
        "hi": "ईमेल भेजने के लिए आपकी पुष्टि चाहिए। ड्राफ़्ट तैयार है।",
        "en": "Sending needs your confirmation. The draft is ready.",
    },
    "UNAUTHORIZED": {
        "kn": "Zoho ಸಂಪರ್ಕ ಮುಗಿದಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಸಂಪರ್ಕಿಸಿ.",
        "hi": "Zoho कनेक्शन समाप्त हो गया है। कृपया दोबारा जोड़ें।",
        "en": "Your Zoho connection has expired. Please reconnect it.",
    },
    "RATE_LIMITED": {
        "kn": "Zoho ಮಿತಿ ಮುಗಿದಿದೆ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "hi": "Zoho की सीमा पूरी हो गई है। कुछ देर बाद कोशिश करें।",
        "en": "Zoho's rate limit is reached. Please try again shortly.",
    },
    "NOT_FOUND": {
        "kn": "ಆ ಮೇಲ್ ಸಿಗಲಿಲ್ಲ. ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಹುಡುಕಿ.",
        "hi": "वह मेल नहीं मिला। कुछ और तरीके से खोजें।",
        "en": "That mail was not found. Try a different search.",
    },
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
    "UNAVAILABLE": {
        "kn": "ಮೇಲ್ ಸೇವೆ ಈಗ ಲಭ್ಯವಿಲ್ಲ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "hi": "मेल सेवा अभी उपलब्ध नहीं है। थोड़ी देर बाद कोशिश करें।",
        "en": "The mail service is not responding right now. Please try again shortly.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "zoho_search_emails",
        "description": "Search the mailbox. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 5},
                "unread_only": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    {
        "name": "zoho_read_thread",
        "description": "Read one thread. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {"thread_id": {"type": "string"}},
            "required": ["thread_id"],
        },
    },
    {
        "name": "zoho_draft_email",
        "description": "Prepare a draft. Does not send; the user confirms before anything leaves.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "cc": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "zoho_send_email",
        "description": "Send a prepared draft. Requires the confirmation token from the draft.",
        "parameters": {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string"},
                "confirmation_token": {"type": "string"},
            },
            "required": ["draft_id", "confirmation_token"],
        },
    },
)

_REQUIRED = {
    "zoho_search_emails": ("query",),
    "zoho_read_thread": ("thread_id",),
    "zoho_draft_email": ("to", "subject", "body"),
    "zoho_send_email": ("draft_id", "confirmation_token"),
}

_READ_ONLY = frozenset({"zoho_search_emails", "zoho_read_thread"})


@dataclass(slots=True)
class _Draft:
    draft_id: str
    session_id: str
    token: str
    expires_at: float
    consumed: bool = False


@dataclass(slots=True)
class MailToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class MailSupervisorResult:
    success: bool
    data: Any = None
    sanitized_output: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_token: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


_drafts: dict[str, _Draft] = {}


def enabled() -> bool:
    return os.getenv("INDOONE_MAIL_SUPERVISOR_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def reset() -> None:
    """Forget every pending draft. A restart does the same, which is the safe direction."""
    _drafts.clear()


def get_tool_declarations() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- the scrubber


def sanitize_mail_body(raw_body: str) -> str:
    """Strip what must not reach a model from a mail body, then bound it.

    An OTP is a credential with a ninety-second life. It is also the single most common
    secret to arrive in a mailbox, and a model asked to summarise the inbox would otherwise
    repeat it into a chat that is stored.
    """
    body = raw_body or ""

    body = _BASE64_IMAGE.sub("[image removed]", body)
    body = _TRACKING_PIXEL.sub("", body)
    body = _PRIVATE_KEY.sub(REDACTED, body)
    body = _PASSWORD_LINE.sub(lambda match: f"{match.group(0).split(':')[0].split('=')[0]}: {REDACTED}", body)
    body = _OTP.sub(lambda match: match.group(0).replace(match.group(1), REDACTED_OTP), body)
    body = _BARE_OTP.sub(lambda match: match.group(0).replace(match.group(1), REDACTED_OTP), body)

    if "<" in body and ">" in body:
        body = _HTML_TAG.sub(" ", body)

    try:
        from app.agents import vault as _vault

        body = _vault().redact(body).redacted_prompt
    except Exception:  # a scrubber that breaks must not hand back the unscrubbed text
        pass

    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if len(body) > MAX_BODY_CHARS:
        kept = body[:MAX_BODY_CHARS]
        cut = kept.rfind("\n")
        kept = kept[:cut] if cut > MAX_BODY_CHARS // 2 else kept
        body = f"{kept}\n\n[Thread truncated for context budget]"

    return body


# ---------------------------------------------------------------- arguments


def validate_arguments(call: MailToolCall) -> tuple[dict[str, Any] | None, str | None]:
    if call.tool_name not in _REQUIRED:
        return None, f"unknown tool {call.tool_name!r}"

    arguments = dict(call.arguments or {})

    for name in _REQUIRED[call.tool_name]:
        value = arguments.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, f"missing argument {name!r}"

    if call.tool_name in {"zoho_draft_email"}:
        for field_name in ("to", "cc"):
            if field_name not in arguments or arguments[field_name] is None:
                continue

            addresses = arguments[field_name]
            if isinstance(addresses, str):
                addresses = [addresses]

            cleaned = [str(address).strip() for address in addresses if str(address).strip()]

            if field_name == "to" and not cleaned:
                return None, "no recipients"

            for address in cleaned:
                if not _EMAIL.match(address):
                    return None, f"invalid email address {address!r}"

            arguments[field_name] = cleaned

    if "limit" in arguments:
        try:
            arguments["limit"] = max(1, min(MAX_SEARCH_RESULTS, int(arguments["limit"])))
        except (TypeError, ValueError):
            arguments["limit"] = 5

    return arguments, None


# ---------------------------------------------------------------- errors


def _error(code: str, language: str) -> MailSupervisorResult:
    table = _MESSAGES.get(code, _MESSAGES["INVALID_ARGUMENT"])
    key = (language or "en").strip().casefold()[:2]

    return MailSupervisorResult(
        success=False,
        error_code=code,
        error_message=table.get(key, table["en"]),
    )


def _status_of(exc: Exception) -> int:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value

    text = str(exc)
    for status in (401, 403, 404, 429, 500, 502, 503, 504):
        if str(status) in text:
            return status

    return 0


def _error_for_status(status: int, language: str) -> MailSupervisorResult:
    if status == 401:
        return _error("UNAUTHORIZED", language)
    if status in {403}:
        return _error("UNAUTHORIZED", language)
    if status == 404:
        return _error("NOT_FOUND", language)
    if status == 429:
        return _error("RATE_LIMITED", language)
    if status >= 500:
        return _error("UNAVAILABLE", language)
    return _error("UNAVAILABLE", language)


# ---------------------------------------------------------------- the gate


def _remember(draft: _Draft, now: float) -> None:
    _drafts[draft.token] = draft

    # Bounded: an expired draft is useless, and a map that grows forever is a leak.
    if len(_drafts) > MAX_PENDING_DRAFTS:
        for token in [token for token, item in _drafts.items() if item.expires_at <= now]:
            _drafts.pop(token, None)
        while len(_drafts) > MAX_PENDING_DRAFTS:
            _drafts.pop(next(iter(_drafts)), None)


def verify_token(draft_id: str, confirmation_token: str, session_id: str, *, now: float | None = None) -> tuple[bool, str]:
    """Whether this token authorises this draft in this session, once."""
    moment = time.monotonic() if now is None else float(now)
    draft = _drafts.get((confirmation_token or "").strip())

    if draft is None:
        return False, "unknown token"
    if draft.consumed:
        return False, "token already used"
    if draft.expires_at <= moment:
        return False, "token expired"
    if draft.draft_id != (draft_id or "").strip():
        return False, "token does not match this draft"
    if draft.session_id != (session_id or "").strip():
        return False, "token does not match this session"

    return True, ""


def verify_and_dispatch(
    draft_id: str,
    confirmation_token: str,
    session_id: str,
    *,
    executor: Callable[..., Any] | None = None,
    language: str = "en",
    now: float | None = None,
) -> MailSupervisorResult:
    """Send a draft, and only when a person has confirmed it.

    This is the only path to a sent message, and it cannot be reached without a token that
    was issued by a draft in this session and has not been used.
    """
    started = time.perf_counter()
    moment = time.monotonic() if now is None else float(now)

    allowed, reason = verify_token(draft_id, confirmation_token, session_id, now=moment)

    if not allowed:
        result = _error("UNAUTHORIZED_MUTATION", language)
        result.sanitized_output = (
            f"{result.error_message} ({reason})" if reason else str(result.error_message)
        )
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    draft = _drafts[confirmation_token.strip()]
    draft.consumed = True

    dispatch = executor or _default_executor()

    try:
        sent = dispatch("zoho_send_email", {"draft_id": draft.draft_id})
    except Exception as exc:  # a failed send is reported, never silently retried
        result = _error_for_status(_status_of(exc), language)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    body = sent if isinstance(sent, str) else json.dumps(sent, ensure_ascii=False, default=str)

    return MailSupervisorResult(
        success=True,
        data=sent,
        sanitized_output=sanitize_mail_body(body),
        action_chips=[],
        requires_confirmation=False,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


# ---------------------------------------------------------------- execution


async def validate_and_execute(
    call: MailToolCall,
    session_id: str,
    token: str | None = None,
    *,
    executor: Callable[..., Any] | None = None,
    language: str = "en",
    now: float | None = None,
) -> MailSupervisorResult:
    """Run a mail tool call under the policy.

    ``token`` is the user's Zoho credential, resolved here and never handed to a model.
    A send is never executed from this path: it needs a confirmation token, and that comes
    from a person pressing a chip, not from a model deciding.
    """
    started = time.perf_counter()
    moment = time.monotonic() if now is None else float(now)

    arguments, problem = validate_arguments(call)

    if problem is not None:
        result = _error("INVALID_ARGUMENT", language)
        result.sanitized_output = f"{result.error_message} ({problem})"
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    assert arguments is not None

    if call.tool_name == "zoho_send_email":
        # The model may ask. It may not send. The token it would need is never in its text.
        return verify_and_dispatch(
            str(arguments["draft_id"]),
            str(arguments["confirmation_token"]),
            session_id,
            executor=executor,
            language=language,
            now=moment,
        )

    run = executor or _default_executor()

    try:
        raw = await run(call.tool_name, arguments, token)
    except Exception as exc:
        result = _error_for_status(_status_of(exc), language)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    body = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    sanitized = sanitize_mail_body(body)

    if call.tool_name == "zoho_draft_email":
        draft_id = _draft_id_from(raw) or f"draft-{secrets.token_hex(4)}"
        confirmation = secrets.token_hex(3)

        _remember(
            _Draft(
                draft_id=draft_id,
                session_id=(session_id or "").strip(),
                token=confirmation,
                expires_at=moment + TOKEN_TTL_SECONDS,
            ),
            moment,
        )

        return MailSupervisorResult(
            success=True,
            data=raw,
            # The model reads the summary. The token is NOT in it: a model that can see the
            # token can send without anyone agreeing, which is the whole failure this gate
            # exists to prevent. The client renders it as a chip instead.
            sanitized_output=(
                f"Draft ready: {arguments['subject']} to {', '.join(arguments['to'])}. "
                "Nothing has been sent; the user must confirm."
            ),
            action_chips=[
                {"label": "Confirm & Send", "action": f"confirm_send:{confirmation}"},
                {"label": "Discard Draft", "action": "discard_draft"},
            ],
            requires_confirmation=True,
            confirmation_token=confirmation,
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    return MailSupervisorResult(
        success=True,
        data=raw,
        sanitized_output=sanitized,
        action_chips=[],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def _draft_id_from(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ("draft_id", "id", "message_id"):
            value = raw.get(key)
            if value:
                return str(value)
    return ""


def _default_executor() -> Callable[..., Any]:
    """The repository's own mail client, behind the credential rule."""

    async def run(tool_name: str, arguments: dict[str, Any], token: str | None) -> Any:
        from app.plugins.email import email_tools

        # The mail client resolves the caller's stored Zoho connection itself; ``token`` is
        # for a caller that already has one (tests, or a channel handler).
        executor = getattr(email_tools, "execute_email_tool", None)

        if executor is None:
            raise RuntimeError("the mail client is not available")

        return await executor(
            user_id=arguments.get("user_id", ""),
            provider="zoho_mail",
            tool_name=tool_name.replace("zoho_", ""),
            arguments=arguments,
            access_token=token,
        )

    return run
