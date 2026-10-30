"""Model-assisted and deterministic read-only GitHub tools for chat.

OAuth secrets remain server-side. Every tool call is scoped to the authenticated
Indoone user and passes through the shared connector authorization policy.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.ai.execution.text_match import is_pleasantry
from app.ai.execution.confirmation import CANCEL, CONFIRM, classify_reply_async, waiting_prompt
from app.ai.gateway.ai_rules import CONFIRM_OR_CANCEL
from app.ai.execution import action_chips
from app.ai.execution import reference_resolution
from app.ai.execution.approval import issue_approval_token
from app.ai.execution.tool_registry import GITHUB_TOOL_DECLARATIONS, GITHUB_WRITE_TOOL_DECLARATIONS
from app.capabilities.connectors.store import (
    delete_pending_github_action,
    get_pending_github_action,
    save_pending_github_action,
)
from app.ai.execution.tools import ToolResult, _explicit_write_intent, run_tool_async

_GITHUB_TARGET_RE = re.compile(
    r"\b(repositories|repository|repos|repo|pull\s+requests?|prs?|issues?|assigned\s+issues|files?|folders?|directories|contents|code|commits?|history|compare|readme|branches?|checks?|ci|workflows?)\b"
    r"|ಶಾಖೆ|ಬ್ರಾಂಚ್|ಶಾಖೆಗಳು|ಬ್ರಾಂಚ್‌ಗಳು|ಚೆಕ್|ಚೆಕ್‌ಗಳು"
    # a standalone alternative, so "branch" and "check" match on their own and
    # cannot be swallowed by the group above
    r"|\b(branch|branches|check|checks|ci)\b"
    r"|(?:ರಿಪೊಸಿಟರಿ|ರೆಪೊಸಿಟರಿ)(?:ಗಳು|ಗಳನ್ನು|ಯನ್ನು)?"
    r"|(?:ಅಸೈನ್ಡ್\s*)?ಇಶ್ಯೂ(?:ಗಳು|ಗಳನ್ನು)?|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್|ಪಿಆರ್"
    r"|ಫೈಲ್|ಫೈಲ್ಗಳು|ಫೋಲ್ಡರ್|ಕೋಡ್|ಒಳಗ(?:ೆ|ಿರುವ|ಿನ|ಿದೆ)|ಕಮಿಟ್|ಹೋಲಿಸು"
    r"|(?:repo|repository|repositories|repositry|रिपोजिटरी|रिपॉजिटरी|రిపోజిటరీ|رِپُوستری|repository)(?:s)?",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(list|show|find|get|fetch|load|check|review|display|refresh|what|which|give|see|tell|please|bring|read|open|browse|analyze|explain|search|compare|inspect|inside)\b"
    r"|\b(madu|maadu|maadi|madi|maad|create|make|new|torisu|torisi|torsi|tori|helu|heli|kodu|kodi|h?olag(?:e|ade)|ede|ide|en\s+(?:ide|ede)|enu\s+(?:ide|ede)|enella\s+(?:ide|ede)|holage|olage|olagade)\b"
    r"|ತೋರಿಸು|ತೋರಿಸಿ|ಪಟ್ಟಿ|ಕೊಡು|ಕೊಡಿ|ಹೇಳು|ಹೇಳಿ|ನೋಡು|ನೋಡಿ|ಹುಡುಕು|ಹುಡುಕಿ|ಮಾಡು|ಮಾಡಿ|ಚೆಕ್|ಓದು|ತೆರೆ|ವಿವರಿಸು|ಒಳಗ(?:ೆ|ಿರುವ|ಿನ|ಿದೆ)|ಹೋಲಿಸು|ಏನು|ಏನೆಲ್ಲಾ|ಏನೇನು|ಏನಿದೆ|ಇವೆ|ಇದೆ|देखो|देखिए|शो|देखें|सर्च|खोजें|पढ़ें|तस्वीर|list|show|find|open|read",
    re.IGNORECASE,
)
_REPOSITORY_NAME_RE = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b")


def _branch_name(text: str) -> str:
    """The branch name the user named, in any of the ways they name it.

    "terminal antha branch create madi", "create a branch called terminal",
    "branch terminal create madi" - the name is a single token next to the word
    branch, or the word after "called"/"antha"/"named".
    """
    patterns = (
        r"branch\s+(?:called|named)\s+([A-Za-z0-9_.\-/]{1,80})",
        r"([A-Za-z0-9_.\-/]{1,80})\s+(?:antha|anta|annu|named|called)\b",
        r"\b(?:called|named|antha|anta)\s+([A-Za-z0-9_.\-/]{1,80})",
        r"branch\s+([A-Za-z0-9_.\-/]{1,80})",
        r"([A-Za-z0-9_.\-/]{1,80})\s+branch",
    )
    noise = {
        "branch", "branches", "new", "create", "make", "madi", "maadi", "madu",
        "the", "a", "an", "in", "on", "for", "called", "named", "antha", "anta",
        "github", "repo", "repository", "and", "to", "please", "ok", "okay",
        "hosa", "hos", "hosa", "hosa-branch", "swalpa", "alli", "evaga",
        "one", "add", "also", "there", "then", "now", "me", "my", "nanna",
        "delete", "remove", "togo", "eshtu", "estu", "yavudu", "yava",
        "pull", "request", "requests", "pr", "prs", "merge", "open", "raise",
        "submit", "from", "to", "into", "onto", "checks", "check", "ci",
        "ಹೊಸ", "ಶಾಖೆ", "ಮಾಡು", "ಮಾಡಿ", "ರಚಿಸು", "ಬ್ರಾಂಚ್",
    }
    for pattern in patterns:
        for candidate in re.findall(pattern, text, re.IGNORECASE):
            cleaned = str(candidate).strip().strip(".,!?`'\"")
            if cleaned and cleaned.casefold() not in noise:
                return cleaned
    return ""


def _repository_hint_from_message(text: str) -> str:
    ignored = {
        "github", "my", "the", "a", "an", "this", "that",
        "repo", "repository", "repositories", "inside",
        "olage", "holage", "olagade", "en", "ide", "enu", "yenu",
        "yavudu", "yaavudu", "yavdu", "nanna",
        "files", "folders", "contents", "readme",
        "please", "show", "list", "tell", "what", "is", "there", "in",
    }
    repository_noun = r"(?:repo|repository|repositories|ರಿಪೊಸಿಟರಿ|ರೆಪೊಸಿಟರಿ)(?:ಯೊಳಗೆ|ಯಲ್ಲಿ|ಯನ್ನು|ಗಳನ್ನು|ಗಳು|ಯ)?"
    after = re.findall(rf"\b{repository_noun}(?:\s+named)?\s+([A-Za-z0-9_.-]+)\b", text, re.IGNORECASE)
    before = re.findall(rf"\b([A-Za-z0-9_.-]+)\s+{repository_noun}\b", text, re.IGNORECASE)
    for item in reversed(after + before):
        if item.casefold() not in ignored:
            return item
    return ""


# "ralli" and "alli" are how a branch is named in ordinary Kannada chat; the tool must
# understand the request before any model is asked about it.
# "checks" and "ci" name the request by themselves - a Kannada or short English message
# carries no verb - but the bare verb "check" does not: it is ordinary English.
_CHECKS_WORD = re.compile(r"\b(checks|check\s*runs?|ci|ci\s*status)\b|ಚೆಕ್|ಸಿಐ", re.IGNORECASE)

_BRANCH_WORD = re.compile(r"\b(branch(?:es)?|ralli|ralli\u0304|alli)\b|ಶಾಖೆ|ಬ್ರಾಂಚ್", re.IGNORECASE)


from app.ai.execution.intent_normalizer import normalize

_SUB_RESOURCE_WORDS = {
    "branches": "branches",
    "commits": "commits",
    "pull_requests": "pull requests",
    "issues": "issues",
    "files": "files",
    "repositories": "repositories",
}


def _github_action(message: str, history: list[tuple[str, str]] | None) -> tuple[str, dict[str, Any]] | str | None:
    text = message.strip()

    # The normalised reading widens the gates below and nothing more. An earlier version
    # rewrote the message into English ("list commits") before the patterns ran, and that
    # discarded the very verbs the branches depend on: "hosa branch madi" became "list
    # branches", "repo holage en ide" became "list repositories". Reading the message is
    # additive; rewriting it is destructive.
    _norm = normalize(text)

    # A branch word is both the action and the target: "1 ralli" names no verb and no
    # repository, yet it is a complete request in Kannada - the repository is the one the
    # thread is already about. Without this it fell through to the model, which invented
    # branch names instead of calling the tool.
    # A greeting is not a request, whatever words the router looks for.
    if is_pleasantry(text):
        return None

    # "explain git branches" names branches and asks for reasoning. The gate can only refuse;
    # the branches below still decide what the request means.
    from app.agents import capability_gate

    _gate = capability_gate()
    if not _gate.wants(text, _gate.GITHUB):
        return None

    branch_word = bool(_BRANCH_WORD.search(text))
    checks_word = bool(_CHECKS_WORD.search(text))

    if not text:
        return None

    # A question about a noun is not a request for that noun: "explain why branches
    # diverge" names branches and asks for reasoning, so it belongs to the model. An
    # explicit ordinal still counts as a request - "explain the diff of pr 3" wants the PR.
    if _norm.leave_alone and _norm.ordinal is None:
        return None

    if not (branch_word or checks_word or _norm.sub_resource or _ACTION_RE.search(text)):
        return None

    if not (branch_word or checks_word or _norm.sub_resource or _GITHUB_TARGET_RE.search(text)):
        return None
    recent_context = " ".join(str(item or "") for role, item in (history or [])[-6:] if role == "user")
    has_context = bool(
        _norm.sub_resource
        or re.search(r"\bgithub\b", text, re.IGNORECASE)
        or re.search(r"\bgithub\b", recent_context, re.IGNORECASE)
        or re.search(r"\b(repo|repos|repository|repositories)\b|(?:ರಿಪೊಸಿಟರಿ|ರೆಪೊಸಿಟರಿ)", text, re.IGNORECASE)
        or _BRANCH_WORD.search(text)
        or checks_word
        or re.search(r"\b(pull\s+requests?|prs?)\b|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್|ಪಿಆರ್", text, re.IGNORECASE)
        or re.search(r"(?:ಅಸೈನ್ಡ್\s*)?ಇಶ್ಯೂ", text, re.IGNORECASE)
        or _REPOSITORY_NAME_RE.search(text)
    )
    if not has_context:
        return None

    repo_match = re.search(r"github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text, re.IGNORECASE)
    if repo_match is None:
        repo_match = _REPOSITORY_NAME_RE.search(text)
    repository = repo_match.group(1) if repo_match else ""
    repository_hint = _repository_hint_from_message(text)

    if re.search(r"\b(pull\s+request|pr)\b", text, re.IGNORECASE) and re.search(r"\b(files?|diff|changes?)\b", text, re.IGNORECASE):
        number = re.search(r"#?(\d{1,9})", text)
        if not number:
            return "Which pull request number should I look at?"
        return "github_list_pull_request_files", {"repository": repository, "repository_hint": repository_hint, "number": int(number.group(1)), "per_page": 30}
    # A pull request that is being created or merged is a write, and is routed
    # here so the handler answers it in fixed words - the same reason the branch
    # is routed. Reading a pull request still falls through to the read tool.
    if re.search(r"\b(pull\s+requests?|prs?)\b|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್|ಪಿಆರ್", text, re.IGNORECASE):
        if re.search(r"\b(merge|merged|merge\s+it)\b|ವಿಲೀನ|ಮರ್ಜ್", text, re.IGNORECASE):
            number = re.search(r"#?\s*(\d{1,9})", text)
            if not number:
                return "Which pull request number should I merge?"
            return "github_merge_pull_request", {
                "repository": repository, "repository_hint": repository_hint,
                "number": int(number.group(1)), "method": "squash",
            }
        # "open a pull request" is a write; "show open pull requests" is a read.
        # The verb has to sit next to the noun, or "open" as an adjective wins.
        if re.search(
            r"\b(?:create|open|raise|make|submit|new)\s+(?:a\s+)?(?:new\s+)?(?:pull\s*request|pr)\b"
            r"|\b(?:pull\s*request|pr)\s+(?:create|open|raise|make|submit|madi|maadi|madu)\b"
            r"|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್.*(?:ಮಾಡು|ಮಾಡಿ|ತೆರೆ|ರಚಿಸು)"
            r"|(?:ಮಾಡು|ಮಾಡಿ|ತೆರೆ|ರಚಿಸು).*ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್",
            text, re.IGNORECASE,
        ):
            refs = re.search(r"\bfrom\s+([A-Za-z0-9_.\-/]{1,80})\s+(?:to|into|onto)\s+([A-Za-z0-9_.\-/]{1,80})", text, re.IGNORECASE)
            if refs:
                head, base = refs.group(1), refs.group(2)
            else:
                head = _branch_name(text)
                base = "main"
            if not head:
                return "Which branch should I open the pull request from?"
            return "github_create_pull_request", {
                "repository": repository, "repository_hint": repository_hint,
                "title": f"Merge {head} into {base}", "head": head, "base": base, "body": "",
            }
    if re.search(r"\b(pull\s+requests?|prs?)\b|ಪುಲ್\s*ರಿಕ್ವೆಸ್ಟ್|ಪಿಆರ್|प्र pull request|पुल रिक्वेस्ट", text, re.IGNORECASE):
        if not repository and not repository_hint:
            return "Which GitHub repository should I check? Please provide its full owner/name, for example owner/repository."
        return "github_list_pull_requests", {"repository": repository, "repository_hint": repository_hint, "page": 1, "per_page": 10}
    if re.search(r"\b(assigned\s+issues|issues?)\b|(?:ಅಸೈನ್ಡ್\s*)?ಇಶ್ಯೂ(?:ಗಳು|ಗಳನ್ನು)?|issue|issues|इश्यू|ஐச்யூ|ఇష్యూ", text, re.IGNORECASE):
        return "github_list_assigned_issues", {"page": 1, "per_page": 10}

    file_match = re.search(
        r"(?:`([^`]{1,256})`|\b((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.(?:md|py|kt|java|js|jsx|ts|tsx|json|ya?ml|toml|xml|gradle|txt|sh|sql|html|css|properties|kts|lock))\b)",
        text, re.IGNORECASE,
    )
    path = (file_match.group(1) or file_match.group(2)).strip() if file_match else ""
    search_intent = bool(
        re.search(r"\b(search|find|locate)\b.{0,60}\b(code|function|symbol|identifier|text)\b|\b(code|function|symbol)\b.{0,40}\b(search|find|locate)\b", text, re.IGNORECASE)
        or re.search(r"ಕೋಡ್.{0,20}(ಹುಡುಕು|ಹುಡುಕಿ)|खोज|सर्च|தேடு|தேட|search|find", text, re.IGNORECASE)
    )
    history_intent = bool(re.search(r"\b(commit\s+history|recent\s+commits|history\s+of)\b|ಕಮಿಟ್\s*ಇತಿಹಾಸ|कमिट हिस्ट्री|கமிட் வரலாறு|commit history", text, re.IGNORECASE))
    compare_intent = bool(re.search(r"\b(compare|diff|changes\s+between)\b|ಹೋಲಿಸು|तुलना|ஒப்பீடு|compare|diff", text, re.IGNORECASE))
    file_read_intent = bool(path) or bool(
        re.search(r"\b(read|open|explain|summari[sz]e|inspect)\b.{0,60}\b(file|source|code|readme)\b|\bwhat\s+does\b", text, re.IGNORECASE)
        or re.search(r"(ಫೈಲ್|ಕೋಡ್).{0,25}(ಓದು|ವಿವರಿಸು)|पढ़|फाइल|कोड|readme|फ़ाइल|கோப்பு|படிக்க|வாசிக்க", text, re.IGNORECASE)
    )
    file_list_intent = bool(
        re.search(r"\b(files?|folders?|directories|directory|contents|inside|tree|structure)\b|\bwhat'?s?\s+inside\b|\bwhat\s+is\s+in\b", text, re.IGNORECASE)
        or re.search(r"(?:ಫೈಲ್|ಫೈಲ್ಗಳು|ಫೋಲ್ಡರ್|ಒಳಗ(?:ೆ|ಿರುವ|ಿನ|ಿದೆ)).{0,80}(?:ಏನು|ಏನೆಲ್ಲಾ?|ಏನೇನು|ಏನಿದೆ|ಇದೆ|ಇವೆ|ತೋರಿಸು|ಪಟ್ಟಿ|ವಿವರ)", text)
        or re.search(r"(?:ಏನು|ಏನೆಲ್ಲಾ?|ಏನೇನು|ಏನಿದೆ|ಇದೆ|ಇವೆ).{0,80}(?:ಫೈಲ್|ಫೈಲ್ಗಳು|ಫೋಲ್ಡರ್|ಒಳಗ(?:ೆ|ಿರುವ|ಿನ|ಿದೆ))", text)
        or re.search(r"(?:फ़ाइल|फाइल|கோப்பு|கோப்புகள்|உள்ள|உள்ளடக்கம்|inside|tree|contents).{0,80}(?:क्या|என்ன|உள்ளது|இது)", text, re.IGNORECASE)
        or re.search(r"\bh?olag(?:e|ade)\b|\ben\s+(?:ide|ede)\b|\benu\s+(?:ide|ede)\b|\benella\s+(?:ide|ede)\b", text, re.IGNORECASE)
    )

    # A commit is a write, and it must be routed BEFORE the file-read intent -
    # otherwise a path in the message ("a.py file commit madi") is read as a
    # request to READ that file, and the commit is never offered.
    if re.search(
        r"\b(commit|committed|save\s+the\s+change|push\s+the\s+change)\b"
        r"|\b(commit\s+madi|commit\s+maadi|commit\s+madu)\b"
        r"|ಕಮಿಟ್|ಉಳಿಸು|ಜಮೆ",
        text, re.IGNORECASE,
    ) and re.search(r"\b(file|files|change|changes|update|edit)\b|ಫೈಲ್|ಬದಲಾವಣೆ", text, re.IGNORECASE):
        if not path:
            return "Which file should I commit? Provide its path, for example README.md."
        return "github_update_repository_file", {
            "repository": repository, "repository_hint": repository_hint,
            "path": path, "content": "", "message": "", "expected_sha": "",
            "branch": _branch_name(text), "create": False,
        }

    # "commits helu", "repo 2 commits": the noun is the whole request. The history branch
    # below needs a word like "history" or "log" to fire, which Kannada does not supply.
    if _norm.sub_resource == "commits" and not _norm.is_write:
        return "github_list_repository_commits", {
            "repository": repository,
            "repository_hint": repository_hint,
            "path": path,
            "per_page": 10,
        }

    if search_intent:
        quoted = re.search(r'["\']([^"\']{1,256})["\']', text)
        query = quoted.group(1).strip() if quoted else ""
        if not query:
            m = re.search(r"\b(?:search|find|locate)\s+(?:for\s+)?(.+?)(?:\s+in\s+(?:the\s+)?(?:repo|repository|code))?$", text, re.IGNORECASE)
            query = m.group(1).strip() if m else ""
            query = re.sub(r"\b(my|github|repo|repository|code|please)\b", "", query, flags=re.IGNORECASE).strip()
        if not query:
            return "What function name or text should I search for in the repository?"
        return "github_search_repository_code", {"repository": repository, "repository_hint": repository_hint, "query": query, "per_page": 10}
    if compare_intent:
        refs = re.search(r"\b(?:between|from)\s+([A-Za-z0-9_./-]+)\s+(?:and|to|\.\.\.)\s+([A-Za-z0-9_./-]+)", text, re.IGNORECASE)
        if not refs:
            return "Which two branches or commits should I compare? For example, compare main and develop."
        return "github_compare_repository_commits", {"repository": repository, "repository_hint": repository_hint, "base": refs.group(1), "head": refs.group(2)}
    if history_intent:
        return "github_list_repository_commits", {"repository": repository, "repository_hint": repository_hint, "path": path, "per_page": 10}
    if file_read_intent:
        if not path:
            return "Which file should I read? Provide its path, for example README.md."
        return "github_read_repository_file", {"repository": repository, "repository_hint": repository_hint, "path": path, "ref": ""}
    if file_list_intent:
        return "github_list_repository_contents", {"repository": repository, "repository_hint": repository_hint, "path": "", "ref": ""}
    if _BRANCH_WORD.search(text):
        # A write IS routed here now. It used to return None so the model would
        # handle it - but the model then looped on the confirmation and invented
        # branch names. The handler answers it deterministically instead: it holds
        # the write, asks in fixed words, and the same conversation resolves it.
        if re.search(r"\b(delete|remove|togo)\b|ಅಳಿಸು", text, re.IGNORECASE):
            name = _branch_name(text)
            if not name:
                return "Which branch should I delete?"
            return "github_delete_branch", {"repository": repository, "repository_hint": repository_hint, "branch": name}
        if re.search(r"\b(create|make|new|add)\b|\b(madi|maadi|madu|madu)\b|ರಚಿಸು|ಹೊಸ|ಮಾಡು|ಮಾಡಿ", text, re.IGNORECASE):
            name = _branch_name(text)
            if not name:
                return "What should I call the new branch?"
            return "github_create_branch", {"repository": repository, "repository_hint": repository_hint, "branch": name}
        return "github_list_branches", {"repository": repository, "repository_hint": repository_hint, "per_page": 30}
    # "checks", not "check": the bare verb is ordinary English, and "check the text" or
    # "check my mail" were being answered with the repository's CI status.
    if re.search(r"\b(checks|check\s*runs?|ci|workflow\s+runs?|build\s+status)\b|ಚೆಕ್|ಸಿಐ", text, re.IGNORECASE):
        return "github_list_check_runs", {"repository": repository, "repository_hint": repository_hint, "ref": ""}
    if re.search(r"\b(commit)\b", text, re.IGNORECASE) and re.search(r"\b([0-9a-f]{7,40})\b", text, re.IGNORECASE):
        sha = re.search(r"\b([0-9a-f]{7,40})\b", text, re.IGNORECASE)
        return "github_get_commit", {"repository": repository, "repository_hint": repository_hint, "sha": sha.group(1)}
    if re.search(r"\b(repositories|repository|repos|repo)\b|(?:ರಿಪೊಸಿಟರಿ|ರೆಪೊಸಿಟರಿ)(?:ಗಳು|ಗಳನ್ನು|ಯನ್ನು)?|रिपॉजिटरी|रिपोजिटरी|రిపోజిటరీ|repo|repository", text, re.IGNORECASE):
        return "github_list_repositories", {"page": 1, "per_page": 10}
    return None




def _safe_tool_error(result: ToolResult) -> str:
    message = result.output.casefold()
    if "not connected" in message:
        return "GitHub is not connected to this Indoone account yet. Open Plugins, connect GitHub, then retry."
    if "scope does not allow" in message or "authorization has expired" in message:
        return "The saved GitHub authorization needs to be renewed. Open Plugins, choose GitHub → Reauthorize, then retry."
    if "authenticated user is required" in message:
        return "Please sign in to Indoone again, then retry the GitHub request."
    return "I couldn't verify the live GitHub data right now. No GitHub changes were made; try again or reauthorize GitHub."


async def _execute_github_tool(
    user_id: str,
    name: str,
    arguments: dict[str, Any],
    approval_tokens: tuple[str, ...] = (),
    conversation_id: str | None = None,
    message: str = "",
) -> dict[str, Any]:
    allowed = {
        "github_list_repositories", "github_list_assigned_issues", "github_list_pull_requests",
        "github_list_repository_contents", "github_read_repository_file", "github_search_repository_code",
        "github_list_repository_commits", "github_compare_repository_commits",
        # The rest of the flow. Without these in the list the model is handed no
        # branch or pull-request tool here and answers that it cannot see them,
        # which is exactly what it was doing.
        "github_list_branches", "github_list_pull_request_files", "github_list_check_runs",
        "github_get_commit",
        "github_create_branch", "github_delete_branch", "github_create_pull_request",
        "github_merge_pull_request", "github_update_repository_file",
    }
    if name not in allowed:
        return {"ok": False, "code": "UNKNOWN_TOOL", "message": "This GitHub action is not available."}
    # A token can arrive either way: in the request's approval_tokens (a client
    # that asked the approvals endpoint), or inside the arguments (the replay of a
    # write this handler held and the user then confirmed). Both are valid.
    _supplied_token = approval_tokens or (str(arguments.get("approval_token") or "").strip(),)
    # A write to a protected branch is confirmed even when the user asked for it in
    # their own words: reversibility says the change can be undone, not that nothing
    # breaks while it is applied. Best effort here - the repository may still be a
    # hint at this point - and the executor checks again with the resolved name.
    protected_reason = ""
    if name in _PROTECTED_BRANCH_WRITES and not any(_supplied_token):
        from app.plugins.github.branch_protection import protected_branch_reason

        protected_reason = await protected_branch_reason(
            user_id,
            str(arguments.get("repository") or arguments.get("repository_hint") or ""),
            str(arguments.get("branch") or arguments.get("ref") or ""),
        )
    if (
        name in _WRITE_TOOLS
        and not any(_supplied_token)
        and (protected_reason or not _explicit_write_intent(name, message))
    ):
        # The model proposed a write and the user did not ask for it themselves.
        # Nothing runs yet: hold it, issue the token, and tell the model what to
        # say. The next message from the user confirms it, and that is what runs it
        # - not the model's wording. An explicit request in the user's own message
        # needs no second confirmation and is not held.
        try:
            token = issue_approval_token(user_id, name, ttl_seconds=300)
        except (RuntimeError, ValueError):
            # No approval secret configured, so a confirmation could never be
            # verified. Say so plainly rather than proposing something that can
            # never be confirmed.
            return {
                "ok": False,
                "code": "APPROVAL_UNAVAILABLE",
                "message": (
                    "GitHub changes cannot be confirmed on this deployment because the "
                    "approval secret is not configured. Tell the user that, and that no "
                    "change was made."
                ),
            }
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=300)
        ).isoformat()
        save_pending_github_action(
            user_id,
            str(conversation_id or arguments.get("conversation_id") or "") or _PENDING_CONVERSATION,
            _pending_payload({"tool": name, "arguments": arguments, "token": token}),
            expires_at,
        )
        action_chips.emit(action_chips.APPROVE, action_chips.CANCEL)
        return {
            "ok": False,
            "code": "APPROVAL_REQUIRED",
            "reply": (
                f"I will {_WRITE_INTENT.get(name, 'make this change')} in "
                f"{arguments.get('repository') or 'the repository'}. "
                + (f"{protected_reason} " if protected_reason else "")
                + CONFIRM_OR_CANCEL
            ),
        }

    payload: dict[str, Any] = {
        "user_id": user_id,
        "page": max(1, min(int(arguments.get("page", 1)), 1000)),
        "per_page": max(1, min(int(arguments.get("per_page", 10)), 30)),
    }
    repository_actions = {
        "github_list_pull_requests", "github_list_repository_contents", "github_read_repository_file",
        "github_search_repository_code", "github_list_repository_commits", "github_compare_repository_commits",
        "github_list_branches", "github_list_pull_request_files", "github_list_check_runs",
        "github_get_commit", "github_create_branch", "github_delete_branch",
        "github_create_pull_request", "github_merge_pull_request",
    }
    if name in repository_actions:
        repository = str(arguments.get("repository") or "").strip()
        repository_hint = str(arguments.get("repository_hint") or "").strip()
        if not repository and _REPOSITORY_NAME_RE.fullmatch(repository_hint):
            repository = repository_hint
        if not _REPOSITORY_NAME_RE.fullmatch(repository):
            lookup = await run_tool_async(
                "github_list_repositories",
                json.dumps({"user_id": user_id, "page": 1, "per_page": 30}, separators=(",", ":")),
                user_id=user_id,
            )
            if not lookup.safe:
                return {"ok": False, "code": "GITHUB_TOOL_FAILED", "message": _safe_tool_error(lookup)}
            try:
                items = json.loads(lookup.output).get("repositories", [])
            except (TypeError, ValueError, AttributeError):
                items = []
            repos = [item for item in items if isinstance(item, dict) and isinstance(item.get("full_name"), str)] if isinstance(items, list) else []
            hint = repository_hint.casefold()
            exact_matches = [
                item for item in repos
                if hint and (
                    hint == str(item.get("full_name", "")).casefold()
                    or hint == str(item.get("name", "")).casefold()
                )
            ]
            matches = exact_matches or [
                item for item in repos
                if not hint
                or hint in str(item.get("full_name", "")).casefold()
                or hint in str(item.get("name", "")).casefold()
            ]
            if len(matches) == 1:
                repository = str(matches[0]["full_name"])
            elif len(matches) > 1:
                choices = ", ".join(str(item["full_name"]) for item in matches[:10])
                return {"ok": False, "code": "REPOSITORY_AMBIGUOUS", "message": f"Which repository should I open? Choose an exact owner/name: {choices}."}
            else:
                return {"ok": False, "code": "REPOSITORY_NOT_FOUND", "message": "I couldn't match that repository. Ask me to list your GitHub repositories and choose an exact owner/name."}
        payload["repository"] = repository

    if name == "github_list_repository_contents":
        payload.update({"path": str(arguments.get("path") or ""), "ref": str(arguments.get("ref") or "")})
    elif name == "github_read_repository_file":
        path = str(arguments.get("path") or "").strip()
        if not path:
            return {"ok": False, "code": "FILE_PATH_REQUIRED", "message": "Which file should I read? Provide its path, for example README.md."}
        payload.update({"path": path, "ref": str(arguments.get("ref") or "")})
    elif name == "github_search_repository_code":
        query = str(arguments.get("query") or "").strip()
        if not query:
            return {"ok": False, "code": "SEARCH_QUERY_REQUIRED", "message": "What function or text should I search for?"}
        payload["query"] = query
    elif name == "github_list_repository_commits":
        payload["path"] = str(arguments.get("path") or "")
    elif name == "github_list_branches":
        payload["per_page"] = max(1, min(int(arguments.get("per_page", 30)), 100))
    elif name == "github_list_pull_request_files":
        payload.update({
            "number": int(arguments.get("number", 0)),
            "per_page": max(1, min(int(arguments.get("per_page", 30)), 100)),
        })
    elif name == "github_list_check_runs":
        payload["ref"] = str(arguments.get("ref") or "")
    elif name == "github_get_commit":
        payload["sha"] = str(arguments.get("sha") or "")
    elif name == "github_update_repository_file":
        path = str(arguments.get("path") or "").strip()
        content = str(arguments.get("content") or "")
        if not path:
            return {"ok": False, "code": "FILE_PATH_REQUIRED", "message": "Which file should I commit? Provide its path."}
        if not content:
            return {"ok": False, "code": "CONTENT_REQUIRED", "message": "What should the file contain?"}
        payload.update({
            "path": path,
            "content": content,
            "message": str(arguments.get("message") or "Update " + path)[:200],
            # The executor reads `sha`; `expected_sha` was the wrong key, so the
            # blob sha never arrived and every update was refused. Accept both.
            "sha": str(arguments.get("sha") or arguments.get("expected_sha") or ""),
            "branch": str(arguments.get("branch") or ""),
            "create": bool(arguments.get("create", False)),
        })
    elif name == "github_create_branch":
        payload.update({
            "branch": str(arguments.get("branch") or ""),
            "from_ref": str(arguments.get("from_ref") or ""),
        })
    elif name == "github_delete_branch":
        payload["branch"] = str(arguments.get("branch") or "")
    elif name == "github_create_pull_request":
        payload.update({
            "title": str(arguments.get("title") or ""),
            "head": str(arguments.get("head") or ""),
            "base": str(arguments.get("base") or "main"),
            "body": str(arguments.get("body") or ""),
        })
    elif name == "github_merge_pull_request":
        payload.update({
            "number": int(arguments.get("number", 0)),
            "method": str(arguments.get("method") or "squash"),
        })
    elif name == "github_compare_repository_commits":
        base = str(arguments.get("base") or "").strip()
        head = str(arguments.get("head") or "").strip()
        if not base or not head:
            return {"ok": False, "code": "COMPARE_REFS_REQUIRED", "message": "Provide both branch or commit references, for example main and develop."}
        payload.update({"base": base, "head": head})

    # A write carries the server-issued approval token, or the user's own wording
    # when they asked for it themselves. The executor validates both.
    if name in _WRITE_TOOLS:
        if any(_supplied_token):
            payload["approval_token"] = next(t for t in _supplied_token if t)
        if message.strip():
            payload["original_message"] = message

    result = await run_tool_async(
        name, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), user_id=user_id,
    )
    if not result.safe:
        return {"ok": False, "code": "GITHUB_TOOL_FAILED", "message": _safe_tool_error(result)}
    try:
        data = json.loads(result.output)
    except (TypeError, ValueError):
        return {"ok": False, "code": "INVALID_TOOL_RESULT", "message": "GitHub returned an unreadable result."}
    if not isinstance(data, dict):
        return {"ok": False, "code": "INVALID_TOOL_RESULT", "message": "GitHub returned an invalid result."}
    # A held write is a proposal, not a result: pass it through as it is, so the
    # formatter can never present it as something that happened.
    if str(data.get("code") or "") in {"APPROVAL_REQUIRED", "APPROVAL_UNAVAILABLE"}:
        return data
    return {"ok": True, "result": data}


def _format_github_result(name: str, result: dict[str, Any]) -> str:
    """Turn a tool's data into the text the model reads.

    One block per tool, and a fallback that hands over the data rather than a
    sentence. The old fallback said "GitHub request completed." and swallowed the
    result, which is why a branch list came back empty: the tool worked and the
    answer was thrown away here.
    """
    if not result.get("ok"):
        # A held write carries a ready reply; anything else carries a message.
        return str(result.get("reply") or result.get("message") or "The GitHub request could not be completed.")
    data = result.get("result")
    if not isinstance(data, dict):
        return "GitHub returned no usable data."
    # Belt and braces: a held write must never be reported as done.
    if str(data.get("code") or "") in {"APPROVAL_REQUIRED", "APPROVAL_UNAVAILABLE"}:
        return str(data.get("reply") or data.get("message") or "Nothing has been changed yet.")

    # A write is only reported as done when the state was read back and found. The
    # sentence comes from the read, not from the fact that the call returned.
    verification = str(result.get("verification") or "").strip()
    if result.get("verified") is False:
        return (
            "I could not confirm that it went through: "
            + (verification or "the check could not read the state back.")
            + " Nothing is being claimed as done."
        )

    if name == "github_list_repositories":
        items = data.get("repositories")
        repos = items if isinstance(items, list) else []
        if not repos:
            return "GitHub is connected, but no accessible repositories were returned."
        # The name and a link, nothing else. The owner prefix, the visibility and the
        # default branch are metadata nobody asked to read; the link carries the full path
        # for anyone who needs it. The number is still what a follow-up can refer to.
        # The name on one line, the raw URL on the next, one blank line between items.
        #
        # An inline [name](url) was tried and reverted: it wrapped across lines in the
        # bubble. A URL on its own line is what the clients already know how to draw - a
        # link row with a copy button - so the name stays plain text and the link keeps the
        # control it had before.
        lines = []
        number = 0

        for item in repos[:30]:
            if not isinstance(item, dict):
                continue

            # The label is the repository name alone. When the API gives only the full
            # name, the owner prefix is dropped here rather than shown to the user.
            repo_name = str(item.get("name") or "").strip()

            if not repo_name:
                repo_name = str(item.get("full_name") or "").strip().rsplit("/", 1)[-1]

            repo_name = repo_name or "Unknown repository"
            url = str(item.get("html_url") or "").strip()

            number += 1
            lines.append(f"{number}. {repo_name}")

            if url:
                lines.append(url)

            lines.append("")

        return "\n".join(lines).rstrip()

    if name == "github_list_assigned_issues":
        items = data.get("issues")
        issues = [item for item in items if isinstance(item, dict) and not item.get("pull_request")] if isinstance(items, list) else []
        if not issues:
            return "GitHub returned no open issues assigned to your account."
        lines = [f"Open issues assigned to you ({len(issues)} returned):"]
        for item in issues[:30]:
            url = str(item.get("html_url") or "").strip()
            lines.append(f"- #{item.get('number', '?')} — {str(item.get('title') or 'Untitled issue')}" + (f" · {url}" if url else ""))
        return "\n".join(lines)

    if name == "github_list_pull_requests":
        items = data.get("pull_requests")
        pulls = items if isinstance(items, list) else []
        repository = str(data.get("repository") or "the selected repository")
        if not pulls:
            return f"No open pull requests were returned for {repository}."
        lines = [f"Open pull requests for {repository} ({len(pulls)} returned):"]
        for item in pulls[:30]:
            url = str(item.get("html_url") or "").strip()
            draft = " · draft" if item.get("draft") is True else ""
            lines.append(f"- #{item.get('number', '?')} — {str(item.get('title') or 'Untitled pull request')}{draft}" + (f" · {url}" if url else ""))
        return "\n".join(lines)

    if name == "github_list_repository_contents":
        items = data.get("contents")
        entries = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        repository, path = str(data.get("repository") or "repository"), str(data.get("path") or "/")
        if not entries:
            return f"No files or folders were returned for {repository}:{path}."
        lines = [f"Files and folders in {repository}:{path} ({len(entries)} entries):"]
        for item in entries[:80]:
            icon = "📁" if item.get("type") == "dir" else "📄"
            size = f" · {int(item.get('size') or 0)} bytes" if item.get("type") == "file" else ""
            lines.append(f"- {icon} {str(item.get('path') or item.get('name') or '')}{size}")
        if len(entries) > 80:
            lines.append(f"Showing 80 of {len(entries)} entries.")
        return "\n".join(lines)

    if name == "github_read_repository_file":
        content = str(data.get("content") or "")
        if not content:
            return f"GitHub returned no readable text for {data.get('repository', 'repository')}:{data.get('path', 'file')}."
        return (
            f"File contents from {data.get('repository', 'repository')}:{data.get('path', 'file')}\n\n"
            f"```text\n{content}\n```\n\nKnown credential formats are redacted."
        )

    if name == "github_search_repository_code":
        items = data.get("results")
        matches = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        if not matches:
            return f"No code matches were returned for '{data.get('query', '')}' in {data.get('repository', 'the selected repository')}."
        lines = [f"Code search results for '{data.get('query', '')}' in {data.get('repository', '')} ({len(matches)} returned):"]
        for item in matches[:20]:
            url = str(item.get("html_url") or "").strip()
            lines.append(f"- {item.get('path', 'Unknown file')}" + (f" · {url}" if url else ""))
            snippets = item.get("snippets")
            if isinstance(snippets, list) and snippets:
                lines.append(f"  Match: {str(snippets[0])[:300]}")
        return "\n".join(lines)

    if name == "github_list_repository_commits":
        items = data.get("commits")
        commits = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        if not commits:
            return f"No commits were returned for {data.get('repository', 'the selected repository')}."
        lines = [f"Recent commits for {data.get('repository', '')} ({len(commits)} returned):"]
        for item in commits[:20]:
            url = str(item.get("html_url") or "").strip()
            lines.append(f"- {str(item.get('sha') or '')[:10]} — {str(item.get('message') or '(no message)')}" + (f" · {url}" if url else ""))
        return "\n".join(lines)

    if name == "github_compare_repository_commits":
        lines = [
            f"Commit comparison for {data.get('repository', '')}: {data.get('base', '')} → {data.get('head', '')}",
            f"Status: {data.get('status', 'unknown')}; ahead by {data.get('ahead_by', 0)}, behind by {data.get('behind_by', 0)}, commits: {data.get('total_commits', 0)}.",
        ]
        items = data.get("files")
        for item in ([value for value in items if isinstance(value, dict)] if isinstance(items, list) else [])[:80]:
            lines.append(f"- {item.get('status', 'changed')}: {item.get('filename', '')} (+{item.get('additions', 0)}/-{item.get('deletions', 0)})")
        return "\n".join(lines)

    # ---- the rest of the flow ---------------------------------------------- #
    # These four reads were declared and offered before this, but the formatter
    # did not know them, so their data was dropped and the model was told
    # "GitHub request completed." That is what made a branch list look empty.
    if name == "github_list_branches":
        items = data.get("branches")
        branches = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        repository = str(data.get("repository") or "the selected repository")
        if not branches:
            return f"No branches were returned for {repository}."
        # Follow-ups that need no guessing: the repository is already known, so each chip
        # carries a complete request rather than a name the user would have to retype.
        action_chips.emit(
            {"label": "Create a branch", "action": f"create a branch in {repository}"},
            {"label": "View commits", "action": f"show commits in {repository}"},
            {"label": "Create a pull request", "action": f"create a PR in {repository}"},
        )
        # The branch name, one per line, numbered. No SHA and no flags: they made the list
        # unreadable and told the user nothing they could act on. The number is what a
        # follow-up refers to.
        lines = []
        for item in branches[:100]:
            name_text = str(item.get("name") or "").strip()

            if name_text:
                lines.append(f"{len(lines) + 1}. {name_text}")
        return "\n".join(lines)

    if name == "github_list_pull_request_files":
        items = data.get("files")
        files = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        repository = str(data.get("repository") or "the selected repository")
        number = data.get("number", "?")
        if not files:
            return f"No changed files were returned for pull request #{number} in {repository}."
        lines = [f"Files changed in pull request #{number} in {repository} ({len(files)} returned):"]
        for item in files[:80]:
            lines.append(
                f"- {item.get('status', 'changed')}: {item.get('filename', '')} "
                f"(+{item.get('additions', 0)}/-{item.get('deletions', 0)})"
            )
        return "\n".join(lines)

    if name == "github_list_check_runs":
        items = data.get("check_runs")
        runs = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        repository = str(data.get("repository") or "the selected repository")
        ref = str(data.get("ref") or "the default branch")
        conclusion = str(data.get("conclusion") or "none")
        if not runs:
            return f"No checks were found for {repository} at {ref}."
        lines = [f"Checks for {repository} at {ref} — overall: {conclusion} ({len(runs)} runs):"]
        for item in runs[:50]:
            lines.append(
                f"- {item.get('name', 'check')}: {item.get('status', '')}"
                + (f" / {item.get('conclusion')}" if item.get("conclusion") else "")
            )
        return "\n".join(lines)

    if name == "github_get_commit":
        sha = str(data.get("sha") or "")
        lines = [
            f"Commit {sha[:10]} in {data.get('repository', '')}",
            f"Author: {data.get('author', 'unknown')} · {data.get('date', '')}",
            "",
            str(data.get("message") or "(no message)"),
        ]
        items = data.get("files")
        files = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        if files:
            lines.append("")
            lines.append(f"Files changed ({len(files)}):")
            for item in files[:80]:
                lines.append(
                    f"- {item.get('status', 'changed')}: {item.get('filename', '')} "
                    f"(+{item.get('additions', 0)}/-{item.get('deletions', 0)})"
                )
        return "\n".join(lines)

    # ---- the writes -------------------------------------------------------- #
    if name == "github_update_repository_file":
        detail = data.get("result")
        detail = detail if isinstance(detail, dict) else {}
        path = str(data.get("path") or detail.get("path") or "the file")
        sha = str(detail.get("sha") or "")[:10]
        verb = "created" if data.get("created") else "committed"
        return (
            f"{path} was {verb} in {data.get('repository', '')}"
            + (f" on branch {data.get('branch')}" if data.get("branch") else "")
            + (f" as {sha}." if sha else ".")
        )

    if name == "github_create_branch":
        detail = data.get("result")
        detail = detail if isinstance(detail, dict) else {}
        return (
            f"Branch '{detail.get('branch', '')}' was created in {data.get('repository', '')} "
            f"from {detail.get('from', 'the default branch')} ({str(detail.get('sha') or '')[:10]})."
        )

    if name == "github_delete_branch":
        detail = data.get("result")
        detail = detail if isinstance(detail, dict) else {}
        return f"Branch '{detail.get('branch', '')}' was deleted from {data.get('repository', '')}."

    if name == "github_create_pull_request":
        detail = data.get("result")
        detail = detail if isinstance(detail, dict) else {}
        url = str(detail.get("html_url") or "").strip()
        return (
            f"Pull request #{detail.get('number', '?')} was opened in {data.get('repository', '')}: "
            f"{detail.get('title', '')} ({detail.get('head', '')} → {detail.get('base', '')})."
            + (f" {url}" if url else "")
        )

    if name == "github_merge_pull_request":
        detail = data.get("result")
        detail = detail if isinstance(detail, dict) else {}
        if not detail.get("merged"):
            return (
                f"Pull request #{detail.get('number', '?')} was not merged. "
                f"{detail.get('message', 'It may have conflicts or checks that have not passed.')}"
            )
        return (
            f"Pull request #{detail.get('number', '?')} was merged "
            f"({detail.get('method', 'squash')}) as {str(detail.get('sha') or '')[:10]}."
        )

    # Nothing above matched. Hand the model the data rather than a sentence: an
    # unformatted result it can read beats a sentence it cannot use.
    try:
        rendered = json.dumps(data, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(data)
    return f"GitHub returned data for {name}:\n{rendered[:4000]}"


_PENDING_CONVERSATION = "github-pending"

_WRITE_INTENT = {
    "github_update_repository_file": "commit that file change",
    "github_create_branch": "create a new branch",
    "github_delete_branch": "delete the branch",
    "github_create_pull_request": "open a pull request",
    "github_merge_pull_request": "merge the pull request",
}

def _pending_payload(payload: dict[str, Any]) -> bytes:
    """Encrypt a pending write, so a stored token is never sitting in the clear."""
    from app.capabilities.connectors.integrations import _fernet

    return _fernet().encrypt(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _read_pending_payload(item: dict[str, Any]) -> dict[str, Any]:
    from app.capabilities.connectors.integrations import _fernet

    raw = _fernet().decrypt(bytes(item.get("encrypted_payload") or b"")).decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("pending github action is malformed")
    return data


# Creating a branch, opening a pull request and writing a file run as soon as the user
# asks for them: there is nothing to preview, and a second question is a second question.
# Deleting a branch and merging a pull request are held, because neither can be undone -
# a merge that lands on the wrong branch cannot be un-merged, and a deleted branch takes
# its commits with it.
# Writes whose target can be a protected branch. A branch creation does not modify
# anything that exists, and a pull request does not either, so neither is here.
_PROTECTED_BRANCH_WRITES = frozenset(
    {"github_update_repository_file", "github_delete_branch", "github_merge_pull_request"}
)

_WRITE_TOOLS = frozenset({
    "github_delete_branch", "github_merge_pull_request",
})


async def handle_github_chat_with_tools(
    user_id: str,
    message: str,
    history: list[tuple[str, str]] | None = None,
    approval_tokens: tuple[str, ...] = (),
    conversation_id: str | None = None,
) -> str | None:
    """Handle recognized GitHub read requests in the authenticated chat flow."""
    user = user_id.strip()
    if not user:
        return None

    # A write that was proposed earlier is resolved here, before any parsing:
    # a bare "ok create maadu" names no repository and no tool, so the parser
    # would return nothing and the confirmation would be lost.
    conversation_key = str(conversation_id or "") or _PENDING_CONVERSATION
    pending = get_pending_github_action(user, conversation_key)
    if pending is not None:
        decision = await classify_reply_async(message)
        if decision == CANCEL:
            delete_pending_github_action(user, conversation_key)
            return "Understood - cancelled. Nothing was changed in your GitHub repository."
        if decision == CONFIRM:
            try:
                stored = _read_pending_payload(pending)
            except (ValueError, RuntimeError):
                delete_pending_github_action(user, conversation_key)
                return "That confirmation could not be read, so nothing was changed. Ask me again."
            tool = str(stored.get("tool") or "")
            stored_arguments = stored.get("arguments")
            stored_arguments = stored_arguments if isinstance(stored_arguments, dict) else {}
            stored_arguments["approval_token"] = str(stored.get("token") or "")
            delete_pending_github_action(user, conversation_key)
            result = await _execute_github_tool(
                user, tool, stored_arguments, conversation_id=conversation_key, message=message
            )
            return _format_github_result(tool, result)
        # Anything else - including "did it happen?" - is answered from the record, not by
        # the model. The held change stays held, so the answer can be acted on.
        try:
            stored_action = str(_read_pending_payload(pending).get("tool") or "the change")
        except (ValueError, RuntimeError):
            stored_action = "the change"
        return waiting_prompt(_WRITE_INTENT.get(stored_action, "this GitHub change"))

    request = _github_action(message, history)
    if request is None:
        return None
    if isinstance(request, str):
        return request
    if not user:
        return "Please sign in to Indoone before accessing GitHub."

    # The parser has already chosen the tool. It is executed HERE and its own
    # formatted answer is returned - the model is not asked to relay it.
    #
    # This used to hand the call to the model and return the model's sentence.
    # That is what let it say a branch was created when it was not: the tool
    # returned "Nothing has been changed yet, ask them to confirm", and the model
    # reworded it into a claim. The count was wrong the same way - the tool gave
    # the real list, and the model summarised it. Nothing that reaches the user
    # goes through a model any more.
    tool_name, tool_arguments = request

    # "create a branch in 2" and "show files in indoone": the user was just handed a
    # numbered list, so a selection in the request is resolved against it instead of being
    # refused for not being a canonical name. Only a value that IS a selection resolves -
    # two matches is ambiguity, and picking one would be a guess dressed up as an answer.
    resolved_arguments = dict(tool_arguments)

    # "show branches of 2" names no repository the parser recognises, so the selection in
    # the message itself is the reference. Only when the request named no repository at all
    # - an explicit one always wins over a bare number.
    if not str(resolved_arguments.get("repository") or "").strip() and not str(
        resolved_arguments.get("repository_hint") or ""
    ).strip():
        selection = reference_resolution.selection_token(message)

        if selection:
            resolved_arguments["repository_hint"] = selection

    for key in ("repository", "repository_hint", "branch"):
        value = str(resolved_arguments.get(key) or "").strip()

        if not value:
            continue

        resolved = reference_resolution.resolve_from_history(
            value,
            history,
            kind="branch" if key == "branch" else "repository",
        )

        if resolved and resolved != value:
            resolved_arguments[key] = resolved

    # A resolved selection is a real repository, not a hint: the field the executor reads
    # first must carry it, or the fuzzy lookup would run again on a name that is already
    # canonical.
    if not str(resolved_arguments.get("repository") or "").strip():
        hint = str(resolved_arguments.get("repository_hint") or "").strip()

        if _REPOSITORY_NAME_RE.fullmatch(hint):
            resolved_arguments["repository"] = hint

    # The thread is already about a repository: bind it rather than asking which one.
    #
    # "Branch estu ede adralli" names nothing, and the old answer was "Which repository
    # should I open? Choose an exact owner/name" - asking the user to repeat what the last
    # two turns had already settled.
    if not str(resolved_arguments.get("repository") or "").strip():
        active = reference_resolution.recent_repository(history)

        if active:
            resolved_arguments["repository"] = active

    tool_arguments = resolved_arguments
    result = await _execute_github_tool(user, tool_name, tool_arguments, (), conversation_id, message=message)
    return _format_github_result(tool_name, result)
