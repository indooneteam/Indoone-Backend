from __future__ import annotations

import re
from typing import Any

_TRIGGER_TYPES = {"comment", "message"}
_ACTION_TYPES = {"reply_comment", "send_message"}
_MAX_RULES = 50
_MAX_RESPONSE_LENGTH = 2200
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")


def _validate_text(value: str, field_name: str, *, required: bool = True) -> str:
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{field_name} is required")
    if len(normalized) > _MAX_RESPONSE_LENGTH:
        raise ValueError(f"{field_name} must be at most 2200 characters")
    return normalized


def _validate_id(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized or not _ID_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def _validate_rule(rule: dict[str, Any]) -> dict[str, object]:
    trigger = str(rule.get("trigger", "")).strip().lower()
    action = str(rule.get("action", "")).strip().lower()
    if trigger not in _TRIGGER_TYPES:
        raise ValueError(f"trigger must be one of: {', '.join(sorted(_TRIGGER_TYPES))}")
    if action not in _ACTION_TYPES:
        raise ValueError(f"action must be one of: {', '.join(sorted(_ACTION_TYPES))}")
    keyword = _validate_text(str(rule.get("keyword", "")), "keyword", required=False).lower()
    response = _validate_text(str(rule.get("response", "")), "response")
    enabled = bool(rule.get("enabled", True))
    return {
        "trigger": trigger,
        "action": action,
        "keyword": keyword,
        "response": response,
        "enabled": enabled,
    }


def _message_matches(text: str, keyword: str) -> bool:
    if not keyword:
        return True
    return keyword in text.lower()


def plan_automation(payload: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    if not 1 <= len(rules) <= _MAX_RULES:
        raise ValueError("rules must contain between 1 and 50 items")

    normalized_rules = [_validate_rule(rule) for rule in rules]
    actions: list[dict[str, object]] = []
    matched_rules: list[int] = []
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        entries = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes", []) if isinstance(entry.get("changes"), list) else []:
            if not isinstance(change, dict):
                continue
            field = str(change.get("field", "")).strip().lower()
            value = change.get("value") if isinstance(change.get("value"), dict) else {}
            if field != "comments":
                continue
            comment_id = str(value.get("id", "")).strip()
            text = str(value.get("text", "")).strip()
            if not comment_id:
                continue
            for index, rule in enumerate(normalized_rules):
                if not rule["enabled"] or rule["trigger"] != "comment":
                    continue
                if not _message_matches(text, str(rule["keyword"])):
                    continue
                matched_rules.append(index)
                actions.append(
                    {
                        "rule_index": index,
                        "action": "reply_comment",
                        "comment_id": _validate_id(comment_id, "comment_id"),
                        "response": str(rule["response"]),
                    }
                )

        for message in entry.get("messaging", []) if isinstance(entry.get("messaging"), list) else []:
            if not isinstance(message, dict):
                continue
            message_body = message.get("message") if isinstance(message.get("message"), dict) else {}
            sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
            text = str(message_body.get("text", "")).strip()
            sender_id = str(sender.get("id", "")).strip()
            if not sender_id:
                continue
            for index, rule in enumerate(normalized_rules):
                if not rule["enabled"] or rule["trigger"] != "message":
                    continue
                if not _message_matches(text, str(rule["keyword"])):
                    continue
                matched_rules.append(index)
                actions.append(
                    {
                        "rule_index": index,
                        "action": "send_message",
                        "recipient_id": _validate_id(sender_id, "recipient_id"),
                        "response": str(rule["response"]),
                    }
                )

    return {
        "integration": "instagram",
        "event_id": payload.get("event_id"),
        "matched_rule_indexes": sorted(set(matched_rules)),
        "actions": actions,
        "dry_run": True,
        "secrets_exposed": False,
    }
