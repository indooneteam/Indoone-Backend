from __future__ import annotations

import json
import re
from typing import Any

from app.ai.service import LocalAIService


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*\`{3}(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\`{3}\s*$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("local AI did not return valid email JSON")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("email draft must be a JSON object")
    return value



async def compose_gmail_email(*, request: str) -> dict[str, object]:
    normalized = request.strip()
    if not normalized:
        raise ValueError("compose request is required")
    if len(normalized) > 20_000:
        raise ValueError("compose request is too long")

    prompt = (
        "Create an email draft from the user's request. Return ONLY a JSON object with "
        'exactly these string fields: "to", "subject", "body". '
        "Do not send the email. Do not invent an email address when the user has not supplied one; "
        "use an empty string for to. Preserve the user's requested language/style. "
        "Never include passwords, OTPs, recovery codes, API keys, card numbers, or other secrets.\n\n"
        f"USER REQUEST:\n{normalized}"
    )
    result = await LocalAIService().generate(prompt, history=[], document_context="")
    data = _extract_json_object(result)

    to = str(data.get("to", "")).strip()
    subject = str(data.get("subject", "")).strip()
    body = str(data.get("body", "")).strip()

    if len(to) > 512 or len(subject) > 2000 or len(body) > 20_000:
        raise ValueError("generated email draft exceeds size limits")
    if not subject or not body:
        raise ValueError("generated email draft must contain subject and body")

    return {
        "operation": "compose",
        "to": to,
        "subject": subject,
        "body": body,
        "send_requires_confirmation": True,
    }
