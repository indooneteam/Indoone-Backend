from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseStyle:
    passed: bool
    reason: str = ""
    checks: tuple[str, ...] = ()


_SENTENCE_RE = re.compile(r"(?<=[.!?।!?])\s+")
_LIST_ITEM_RE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+")


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def assess_response_style(
    prompt: str,
    response: str,
    *,
    profile: str = "",
) -> ResponseStyle:
    """Check explicit style constraints without judging writing quality."""

    text = response.strip()
    normalized_prompt = _normalized(prompt)
    normalized_response = _normalized(response)

    if not text:
        return ResponseStyle(False, "style_empty_response")

    checks: list[str] = []
    style = profile.casefold().strip()

    concise_requested = "concise" in normalized_prompt or style == "concise"
    exact_three = "exactly three" in normalized_prompt or style == "exactly_three_steps"
    bullets_requested = any(marker in normalized_prompt for marker in ("bullet points", "bulleted list", "bullet list"))

    if concise_requested:
        sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
        if len(sentences) > 5 or len(text) > 600:
            return ResponseStyle(False, "style_not_concise", tuple(checks))
        checks.append("concise")

    if exact_three:
        items = _LIST_ITEM_RE.findall(response)
        if len(items) != 3:
            return ResponseStyle(False, "style_exactly_three_items", tuple(checks))
        checks.append("exactly_three_items")

    if bullets_requested and not _LIST_ITEM_RE.search(response):
        return ResponseStyle(False, "style_bullets_missing", tuple(checks))
    if bullets_requested:
        checks.append("bullets")

    if "all lowercase" in normalized_prompt and response != response.lower():
        return ResponseStyle(False, "style_lowercase_required", tuple(checks))

    if normalized_prompt and "no markdown" in normalized_prompt and any(mark in response for mark in ("**", "##", "`")):
        return ResponseStyle(False, "style_markdown_not_allowed", tuple(checks))

    return ResponseStyle(True, "", tuple(checks))
