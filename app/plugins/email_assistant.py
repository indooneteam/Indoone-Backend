from __future__ import annotations

import base64
import logging
import json
import re
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any

from app.ai.gateway.service import generate_reply
from app.capabilities.integrations import _fernet
from app.capabilities.store import (
    delete_pending_email_action,
    get_integration_token_metadata,
    get_pending_email_action,
    save_pending_email_action,
)
from app.plugins.zoho_mail.service import (
    delete_message,
    get_message_content,
    list_accounts,
    list_folders,
    list_messages,
    replace_draft,
    reply_to_message,
    save_draft,
    send_message,
)

from app.ai.language.language_detection import (
    INDIC_CANCEL_WORDS,
    INDIC_CONFIRM_WORDS,
    INDIC_MAIL_WORDS,
    indic_pattern,
)

logger = logging.getLogger(__name__)

MAX_EMAIL_RESULTS = 10
EMAIL_ADDRESS_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
# The Indian-language words come from one shared table in app/ai/language_detection.py,
# so a confirmation typed in Tamil and one typed in Kannada are recognised by the same
# list instead of by hand-written patterns that only ever grew in one language. The
# Kannada extras that were here before are kept alongside it.
_INDIC_MAIL = indic_pattern(INDIC_MAIL_WORDS)
_INDIC_CONFIRM = indic_pattern(INDIC_CONFIRM_WORDS)
_INDIC_CANCEL = indic_pattern(INDIC_CANCEL_WORDS)

EMAIL_CONTEXT_RE = re.compile(
    r"\b(email|emails|mail|mails|inbox|zoho|draft|drafts|unread)\b|"
    + _INDIC_MAIL
    + r"|ಡ್ರಾಫ್ಟ್|ಜಿಮೇಲ್|ಜೋಹೋ|ಜೂಹೋ",
    re.IGNORECASE,
)
CONFIRM_RE = re.compile(
    r"^(yes|yes please|confirm|confirmed|do it|go ahead|send it|delete them|"
    r"approve|approved|"
    + _INDIC_CONFIRM
    + r"|ಮಾಡು|ಡಿಲೀಟ್ ಮಾಡು)[.! ]*$",
    re.IGNORECASE,
)
CANCEL_RE = re.compile(
    r"^(no|no thanks|cancel|stop|don't|do not|"
    + _INDIC_CANCEL
    + r"|ಕ್ಯಾನ್ಸಲ್)[.! ]*$",
    re.IGNORECASE,
)


def _rows(payload: object, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    candidate_keys = tuple(dict.fromkeys(("data", *keys, "accounts", "messages", "folders", "drafts")))
    for key in candidate_keys:
        nested = payload.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        if isinstance(nested, dict):
            rows = _rows(nested, *keys)
            if rows:
                return rows
    return []


def _zoho_accounts(payload: dict[str, Any]) -> list[dict[str, str]]:
    accounts: list[dict[str, str]] = []
    for row in _rows(payload, "accounts"):
        account_id = str(row.get("accountId") or row.get("account_id") or "").strip()
        address = str(
            row.get("primaryEmailAddress")
            or row.get("mailboxAddress")
            or row.get("incomingUserName")
            or ""
        ).strip()
        if account_id:
            accounts.append({"id": account_id, "email": address})
    return accounts


def _format_zoho_timestamp(value: Any) -> str:
    """Normalize Zoho epoch-millisecond values to readable UTC timestamps."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.isdigit():
        try:
            numeric = int(raw)
            is_milliseconds = numeric > 100_000_000_000
            timestamp = numeric / 1000 if is_milliseconds else numeric
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            precision = "milliseconds" if is_milliseconds else "seconds"
            return parsed.isoformat(timespec=precision).replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError):
            return raw
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        return raw


def _zoho_summary(
    row: dict[str, Any],
    account_id: str,
    default_folder_id: str = "",
) -> dict[str, str]:
    # Prefer the actual address field over a display-name-only "sender" field;
    # replies need a verified recipient and user summaries should be unambiguous.
    sender = (
        row.get("fromAddress")
        or row.get("from_address")
        or row.get("from")
        or row.get("sender")
        or "Unknown sender"
    )
    if isinstance(sender, dict):
        sender = (
            sender.get("emailAddress")
            or sender.get("email")
            or sender.get("address")
            or sender.get("name")
            or "Unknown sender"
        )
    received_time = (
        row.get("receivedTime")
        or row.get("receivedtime")
        or row.get("receivedTimeInGMT")
        or row.get("receivedtimeingmt")
    )
    sent_time = (
        row.get("sentTime")
        or row.get("senttime")
        or row.get("sentDateInGMT")
        or row.get("sentdateingmt")
        or row.get("sentDate")
    )
    return {
        "id": str(row.get("messageId") or row.get("message_id") or row.get("id") or "").strip(),
        "account_id": account_id,
        "folder_id": str(row.get("folderId") or row.get("folder_id") or default_folder_id).strip(),
        "subject": str(row.get("subject") or "(No subject)").strip(),
        "from": str(sender).strip(),
        "date": _format_zoho_timestamp(received_time or sent_time or row.get("date") or ""),
        "to": str(row.get("toAddress") or row.get("to_address") or row.get("to") or "").strip(),
        "snippet": str(row.get("summary") or row.get("snippet") or row.get("content") or "").strip(),
        "body": "",
    }

def _encrypt_payload(payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _fernet().encrypt(raw)


def _decrypt_payload(payload: bytes) -> dict[str, Any]:
    value = json.loads(_fernet().decrypt(bytes(payload)).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("pending action payload is invalid")
    return value


def _save_pending(
    user_id: str,
    conversation_id: str,
    provider: str,
    action_type: str,
    payload: dict[str, Any],
) -> None:
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    save_pending_email_action(
        user_id,
        conversation_id,
        provider,
        action_type,
        _encrypt_payload(payload),
        expires_at,
    )


def _provider_requested(message: str) -> str:
    lowered = message.casefold()
    if "zoho" in lowered or "ಜೋಹೋ" in lowered or "ಜೂಹೋ" in lowered:
        return "zoho_mail"
    return ""


def _is_recent(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in (
        "recent", "latest", "new mail", "new email", "unread", "inbox",
        "ಹೊಸ ಮೇಲ್", "ಹೊಸದಾಗಿ", "ಇತ್ತೀಚಿನ", "ಮೇಲ್ ಬಂದಿದೆಯಾ",
    ))


def _is_delete(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in ("delete", "remove", "trash", "ಡಿಲೀಟ್", "ತೆಗೆದು ಹಾಕು", "ಅಳಿಸು"))


def _is_send(message: str) -> bool:
    lowered = message.casefold()
    # Match an actual send intent, not substrings such as "send" inside "sender".
    english_words = set(re.findall("[a-z]+", lowered))
    english_send_intent = bool({"send", "sending"} & english_words)
    return english_send_intent or any(
        phrase in lowered for phrase in ("ಕಳುಹಿಸು", "ಸೆಂಡ್ ಮಾಡು")
    )


def _is_draft(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in ("draft", "compose", "write an email", "write a mail", "ಡ್ರಾಫ್ಟ್", "ಬರೆಯು"))


def _is_read(message: str) -> bool:
    lowered = message.casefold()
    return any(word in lowered for word in ("read this", "read the email", "open email", "email content", "ಓದಿ", "ಏನಂತಿದೆ"))


def _search_query(message: str) -> str:
    lowered = message.casefold()
    if "finance" in lowered or "financial" in lowered:
        return "finance"
    for marker in ("about ", "subject ", "from ", "containing ", "named "):
        index = lowered.find(marker)
        if index >= 0:
            value = message[index + len(marker):]
            value = re.sub(
                r"\b(please|can you|could you|in my|in the|my|the|emails?|mails?|inbox|zoho)\b",
                " ",
                value,
                flags=re.IGNORECASE,
            )
            value = " ".join(value.split()).strip(" ?!.,\"'")
            if value:
                return value[:120]
    cleaned = re.sub(
        r"\b(please|find|search|show|look for|check|read|open|emails?|mails?|inbox|my|the|zoho|recent|latest|new|delete|remove|trash)\b",
        " ",
        message,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split()).strip(" ?!.,\"'")[:120]


def _mail_timestamp(value: object) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    if raw.isdigit():
        number = int(raw)
        # Zoho may return epoch milliseconds or seconds.
        if number > 10_000_000_000:
            number = number / 1000
        return float(number)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        return 0.0


async def _list_mail(
    user_id: str,
    provider: str,
    query: str,
    recent: bool = False,
    limit: int = MAX_EMAIL_RESULTS,
) -> list[dict[str, str]]:
    limit = max(1, min(int(limit), 200))

    accounts = _zoho_accounts(await list_accounts(user_id))
    if not accounts:
        raise RuntimeError("Zoho Mail did not return a mailbox account")
    account = accounts[0]

    if recent:
        # Avoid Zoho's newMails search filter for "latest" requests; resolve
        # the actual Inbox and page its messages instead.
        folder_payload = await list_folders(user_id, account["id"])
        folders = _rows(folder_payload, "folders")
        inbox = next(
            (
                row for row in folders
                if str(row.get("folderName") or row.get("name") or "").strip().casefold() == "inbox"
                or str(row.get("folderType") or "").strip().casefold() == "inbox"
            ),
            None,
        )
        inbox_id = str((inbox or {}).get("folderId") or (inbox or {}).get("id") or "").strip()
        if not inbox_id:
            raise RuntimeError("Zoho Mail Inbox folder could not be identified; refusing an unscoped mailbox listing")
        listed = await list_messages(
            user_id,
            account["id"],
            folder_id=inbox_id,
            start=1,
            limit=limit,
        )
        rows = [
            _zoho_summary(row, account["id"], inbox_id)
            for row in _rows(listed, "messages")
        ]
    else:
        listed = await list_messages(
            user_id,
            account["id"],
            search_key=query.strip(),
            start=1,
            limit=limit,
        )
        rows = [_zoho_summary(row, account["id"]) for row in _rows(listed, "messages")]

    # Providers normally return newest-first, but normalise ordering when
    # their payload includes parseable received/sent timestamps.
    if any(_mail_timestamp(row.get("date")) for row in rows):
        rows.sort(key=lambda row: _mail_timestamp(row.get("date")), reverse=True)
    return rows


async def _read_body(user_id: str, provider: str, row: dict[str, str]) -> str:
    details = await get_message_content(user_id, row["account_id"], row["folder_id"], row["id"])
    data = details.get("data")
    if isinstance(data, dict):
        details = data
    body = details.get("content") or details.get("body") or details.get("messageContent") or ""
    return str(body).strip()[:5000]


def _format_results(rows: list[dict[str, str]], label: str) -> str:
    if not rows:
        return "No matching emails were returned from " + label + "."
    lines = ["Found " + str(len(rows)) + " email(s) in " + label + ":"]
    for index, row in enumerate(rows, start=1):
        line = str(index) + ". " + row.get("subject", "(No subject)")
        line += " — From: " + row.get("from", "Unknown sender")
        if row.get("date"):
            line += " — " + row["date"]
        if row.get("snippet"):
            line += "\n   " + row["snippet"][:180]
        lines.append(line)
    return "\n".join(lines)


async def _compose_fields(user_message: str, recipient: str) -> tuple[str, str]:
    prompt = (
        "Write a polished email based on the user's request. Do not send it. "
        "Return only this format:\nSUBJECT: <short subject>\nBODY:\n<email body>\n"
        "Recipient address: " + recipient + "\nUser request: " + user_message[:4000]
    )
    generated = (await generate_reply(prompt)).strip()
    subject_match = re.search(r"(?im)^SUBJECT:\s*(.+)$", generated)
    body_match = re.search(r"(?is)^.*?BODY:\s*(.*)$", generated)
    subject = subject_match.group(1).strip() if subject_match else "Email draft"
    body = body_match.group(1).strip() if body_match else generated
    fence = chr(96) * 3
    body = body.removeprefix(fence + "text").removeprefix(fence + "markdown").removeprefix(fence).removesuffix(fence).strip()
    if not body:
        raise RuntimeError("The AI could not prepare an email draft. Please try again.")
    return subject[:2000], body[:20000]


async def _draft_or_confirm_send(
    user_id: str,
    conversation_id: str,
    provider: str,
    request: str,
    recipient: str,
    sending: bool,
) -> str:
    subject, body = await _compose_fields(request, recipient)
    if sending:
        payload: dict[str, Any] = {"to": recipient, "subject": subject, "body": body}
        if provider == "zoho_mail":
            accounts = _zoho_accounts(await list_accounts(user_id))
            if not accounts or not accounts[0]["email"]:
                raise RuntimeError("Zoho Mail did not return the sender email address needed to send.")
            payload["account_id"] = accounts[0]["id"]
            payload["from_address"] = accounts[0]["email"]
        _save_pending(user_id, conversation_id, provider, "send_email", payload)
        return (
            "Prepared an email to " + recipient + ".\nSubject: " + subject + "\n\n" + body
            + "\n\nI have not sent it yet. Reply CONFIRM to send, or CANCEL."
        )
    accounts = _zoho_accounts(await list_accounts(user_id))
    if not accounts or not accounts[0]["email"]:
        raise RuntimeError("Zoho Mail did not return the sender email address needed to save a draft.")
    await save_draft(user_id, accounts[0]["id"], accounts[0]["email"], recipient, subject, body)
    return "Draft saved in " + "Zoho Mail" + ".\nSubject: " + subject


def _is_edit_draft(message: str) -> bool:
    lowered = message.casefold()
    has_draft = "draft" in lowered or "ಡ್ರಾಫ್ಟ್" in lowered
    has_edit = any(word in lowered for word in ("edit", "update", "revise", "modify", "change", "ಬದಲಿಸು", "ಎಡಿಟ್"))
    return has_draft and has_edit and not _is_delete(message)


async def _list_drafts(user_id: str, provider: str) -> list[dict[str, str]]:
    accounts = _zoho_accounts(await list_accounts(user_id))
    if not accounts:
        return []
    account = accounts[0]
    folder_payload = await list_folders(user_id, account["id"])
    folders = _rows(folder_payload, "folders")
    draft_folder = next(
        (
            row for row in folders
            if str(row.get("folderName") or row.get("name") or "").casefold() == "drafts"
            or str(row.get("folderType") or "").casefold() == "drafts"
        ),
        None,
    )
    if not draft_folder:
        return []
    folder_id = str(draft_folder.get("folderId") or draft_folder.get("id") or "").strip()
    if not folder_id:
        return []
    listed = await list_messages(user_id, account["id"], folder_id=folder_id, start=1, limit=MAX_EMAIL_RESULTS)
    rows = []
    for raw in _rows(listed, "messages"):
        row = _zoho_summary(raw, account["id"])
        row["folder_id"] = row["folder_id"] or folder_id
        row["draft_id"] = row["id"]
        row["from"] = str(raw.get("fromAddress") or account["email"] or "")
        rows.append(row)
    return rows


def _draft_query(message: str) -> str:
    quoted = re.search(r"""["']([^"']{1,120})["']""", message)
    if quoted:
        return quoted.group(1).strip()
    value = _search_query(message)
    value = re.sub(r"\b(edit|update|revise|modify|change|draft|drafts|email|emails|mail|mails|please|my|the|to say|as)\b", " ", value, flags=re.IGNORECASE)
    return " ".join(value.split()).strip(" ?!.,\"'")[:120]


async def _edit_draft(
    user_id: str,
    provider: str,
    message: str,
) -> str:
    rows = await _list_drafts(user_id, provider)
    if not rows:
        return "I couldn't find any drafts in " + "Zoho Mail" + "."
    query = _draft_query(message)
    matches = [
        row for row in rows
        if not query or query.casefold() in row.get("subject", "").casefold()
        or query.casefold() in row.get("snippet", "").casefold()
    ]
    if len(matches) != 1:
        return "Please identify the draft by subject. Available drafts:\n" + _format_results(rows[:MAX_EMAIL_RESULTS], provider)
    row = matches[0]
    old_body = row.get("body", "")
    if not old_body:
        old_body = await _read_body(user_id, provider, row)
    recipient = row.get("to", "")
    instructions = (
        "Revise this existing email draft according to the user's requested changes. "
        "Preserve details not asked to change. Return SUBJECT and BODY only.\n"
        "Existing subject: " + row.get("subject", "") + "\n"
        "Existing recipient: " + recipient + "\n"
        "Existing body: " + old_body[:8000] + "\n"
        "Requested edits: " + message[:4000]
    )
    subject, body = await _compose_fields(instructions, recipient)
    accounts = _zoho_accounts(await list_accounts(user_id))
    if not accounts or not accounts[0]["email"] or not row.get("folder_id") or not row.get("draft_id"):
        return "I found the Zoho draft but could not verify the folder/account needed to safely replace it. Nothing was changed."
    await replace_draft(
        user_id,
        accounts[0]["id"],
        row["folder_id"],
        row["draft_id"],
        accounts[0]["email"],
        recipient,
        subject,
        body,
        approved=True,
    )
    return "Updated the Zoho Mail draft: " + subject


async def _delete_drafts(
    user_id: str,
    conversation_id: str,
    provider: str,
    message: str,
) -> str:
    rows = await _list_drafts(user_id, provider)
    if not rows:
        return "No drafts were found in " + "Zoho Mail" + "."
    query = _draft_query(message)
    matches = [
        row for row in rows
        if not query or query.casefold() in row.get("subject", "").casefold()
        or query.casefold() in row.get("snippet", "").casefold()
    ]
    if not matches:
        return "No draft matched that subject. Nothing was deleted."
    items = []
    for row in matches:
        items.append({
            "id": row.get("id", ""),
            "draft_id": row.get("draft_id", ""),
            "account_id": row.get("account_id", ""),
            "folder_id": row.get("folder_id", ""),
        })
    _save_pending(user_id, conversation_id, provider, "delete_drafts", {"items": items})
    preview = _format_results(matches, "Zoho Mail drafts")
    return preview + "\n\nNothing has been deleted yet. Reply CONFIRM to delete these draft(s), or CANCEL."


async def _execute_pending(user_id: str, provider: str, action_type: str, payload: dict[str, Any]) -> str:
    if action_type == "send_reply":
        if provider != "zoho_mail":
            raise ValueError("reply confirmation is only supported for Zoho Mail")
        result = await reply_to_message(
            user_id,
            str(payload["account_id"]),
            str(payload["message_id"]),
            str(payload["from_address"]),
            str(payload["to"]),
            str(payload["subject"]),
            str(payload["body"]),
            approved=True,
        )
        status = result.get("status") if isinstance(result, dict) else None
        data = result.get("data") if isinstance(result, dict) else None
        status_code = status.get("code") if isinstance(status, dict) else None
        if not isinstance(status, dict) or str(status_code) not in {"200", "201", "202"}:
            raise RuntimeError("Zoho Mail did not return a verified successful reply status")
        if not isinstance(data, dict) or not str(data.get("messageId") or "").strip():
            raise RuntimeError("Zoho Mail reply response did not include a verifiable message ID")
        logger.info(
            "confirmed Zoho Mail reply verified by provider",
            extra={"provider": provider, "provider_message_id_present": True},
        )
        return "Reply sent successfully through Zoho Mail."

    if action_type == "send_email":
        result = await send_message(
            user_id,
            str(payload["account_id"]),
            str(payload["from_address"]),
            str(payload["to"]),
            str(payload["subject"]),
            str(payload["body"]),
            approved=True,
        )
        status = result.get("status") if isinstance(result, dict) else None
        data = result.get("data") if isinstance(result, dict) else None
        status_code = status.get("code") if isinstance(status, dict) else None
        if not isinstance(status, dict) or str(status_code) not in {"200", "201", "202"}:
            raise RuntimeError("Zoho Mail did not return a verified successful send status")
        if not isinstance(data, dict) or not str(data.get("messageId") or "").strip():
            raise RuntimeError("Zoho Mail send response did not include a verifiable message ID")

        logger.info(
            "confirmed email send verified by provider",
            extra={"provider": provider, "provider_message_id_present": True},
        )
        return "Email sent successfully through " + "Zoho Mail" + "."

    if action_type == "delete_messages":
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return "No saved email targets were available to delete."
        completed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            await delete_message(
                user_id,
                str(item["account_id"]),
                str(item["folder_id"]),
                str(item["id"]),
                approved=True,
                permanently=False,
            )
            completed += 1
        return "Moved " + str(completed) + " email(s) to Trash."

    if action_type == "delete_drafts":
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            return "No saved draft targets were available to delete."
        completed = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            await delete_message(
                user_id,
                str(item["account_id"]),
                str(item["folder_id"]),
                str(item["draft_id"] or item["id"]),
                approved=True,
                permanently=False,
            )
            completed += 1
        return "Deleted " + str(completed) + " draft(s)."

    return "This email action is no longer supported. Please ask again."


async def handle_email_chat(user_id: str, conversation_id: str, message: str) -> str | None:
    """Run mail-specific tools for clear requests; return None for normal chat."""
    user = user_id.strip()
    conversation = conversation_id.strip()
    text = message.strip()
    if not user or not conversation or not text:
        return None

    pending = get_pending_email_action(user, conversation)
    if pending:
        try:
            data = _decrypt_payload(bytes(pending["encrypted_payload"]))
        except Exception:
            delete_pending_email_action(user, conversation)
            return "The pending email action could not be recovered. Please ask again."
        action_type = str(pending["action_type"])
        provider = str(pending["integration_id"])
        if action_type == "await_recipient":
            recipient_match = EMAIL_ADDRESS_RE.search(text)
            if recipient_match:
                delete_pending_email_action(user, conversation)
                return await _draft_or_confirm_send(
                    user, conversation, provider, str(data["request"]),
                    recipient_match.group(0), bool(data.get("sending")),
                )
            if CANCEL_RE.fullmatch(text):
                delete_pending_email_action(user, conversation)
                return "Cancelled. Nothing was sent or saved."
            return "Please reply with the recipient email address, or CANCEL."
        if CONFIRM_RE.fullmatch(text):
            delete_pending_email_action(user, conversation)
            try:
                return await _execute_pending(user, provider, action_type, data)
            except Exception as exc:
                logger.warning(
                    "confirmed email action could not be verified",
                    extra={
                        "provider": provider,
                        "action_type": action_type,
                        "exception_type": type(exc).__name__,
                    },
                )
                if action_type == "send_email":
                    return (
                        "I couldn't verify that the email was accepted by the mail provider. "
                        "Check the provider's Sent folder before retrying to avoid sending a duplicate."
                    )
                return "The approved email action could not be verified. Check the mailbox before retrying."
        if CANCEL_RE.fullmatch(text):
            delete_pending_email_action(user, conversation)
            return "Cancelled. No email was sent or deleted."
        delete_pending_email_action(user, conversation)

    # Do not hijack ordinary questions containing words such as "latest"
    # or "research". Require an explicit email/mailbox/provider reference.
    if not EMAIL_CONTEXT_RE.search(text):
        return None

    connected = []
    for provider in ("zoho_mail",):
        if get_integration_token_metadata(user, provider) is not None:
            connected.append(provider)
    requested = _provider_requested(text)
    if requested:
        if requested not in connected:
            label = "Zoho Mail"
            return label + " is not connected to this Indoone account yet. Connect it from Plugins first."
        provider = requested
    else:
        if not connected:
            return "No email account is connected to this Indoone account. Connect Email or Zoho Mail from Plugins first."
        provider = connected[0]

    label = "Zoho Mail"
    recipient_match = EMAIL_ADDRESS_RE.search(text)
    recipient = recipient_match.group(0) if recipient_match else ""

    if _is_delete(text) and "draft" in text.casefold():
        return await _delete_drafts(user, conversation, provider, text)

    if _is_edit_draft(text):
        return await _edit_draft(user, provider, text)

    if _is_send(text) or _is_draft(text):
        sending = _is_send(text)
        if not recipient:
            _save_pending(user, conversation, provider, "await_recipient", {"request": text, "sending": sending})
            return "What recipient email address should I use? I will not send anything without showing you the draft and getting confirmation."
        return await _draft_or_confirm_send(user, conversation, provider, text, recipient, sending)

    if _is_delete(text):
        query = _search_query(text)
        all_request = bool(re.search(r"\b(all|every|everything)\b|ಎಲ್ಲಾ|ಎಲ್ಲ", text, re.IGNORECASE))
        if all_request and (not query or query.casefold() in {"email", "mail", "emails"}):
            return "For safety, specify a filter such as a sender, subject, Finance, or date range. I will list matches and ask you to confirm before moving them to Trash."
        if not query:
            return "Which email should I delete? Tell me the sender, subject, or search term."
        rows = await _list_mail(user, provider, query)
        rows = [row for row in rows if row.get("id")]
        if not rows:
            return "No emails matched '" + query + "' in " + label + ". Nothing was deleted."
        items = [
            {"id": row["id"], "account_id": row.get("account_id", ""), "folder_id": row.get("folder_id", "")}
            for row in rows
        ]
        _save_pending(user, conversation, provider, "delete_messages", {"items": items, "query": query})
        return (
            _format_results(rows, label)
            + "\n\nI will move these " + str(len(items))
            + " matching email(s) to Trash. Nothing has been deleted yet. Reply CONFIRM or CANCEL."
        )

    recent = _is_recent(text)
    query = _search_query(text)
    rows = await _list_mail(user, provider, text if recent else query, recent=recent)
    if _is_read(text):
        if not rows:
            return "No matching emails were found in " + label + "."
        if len(rows) > 1:
            return "I found multiple matching emails. Add the sender or subject to narrow it down:\n" + _format_results(rows[:5], label)
        body = await _read_body(user, provider, rows[0])
        if not body:
            return "I found '" + rows[0].get("subject", "(No subject)") + "', but the provider returned no readable body."
        return "Email: " + rows[0].get("subject", "(No subject)") + "\nFrom: " + rows[0].get("from", "Unknown sender") + "\n\n" + body
    return _format_results(rows, label)
