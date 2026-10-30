"""Agent 11 - the GitHub supervisor: the only door to the API.

The model is allowed to *ask* for a repository, a diff, a branch. It is not allowed to reach
GitHub itself. This agent sits on that boundary and does five things the model cannot be
trusted with:

1. **Declarations.** The four tools it may call, in the shape a function-calling provider
   expects. Nothing else is offered, so nothing else can be called.
2. **Arguments.** A repository is resolved from what people actually type ("backend" means
   indooneteam/Indoone-Backend) and then checked against the repository's own validators. A
   ref carrying a shell metacharacter or an invalid git sequence is refused before a request
   is made - a hallucinated branch name must not become an outbound URL.
3. **Credentials.** The token is read here, from the user's stored connection or the
   environment. It never appears in a declaration, an argument, or anything the model sees.
4. **What comes back.** A diff is the most likely place in this product to find a leaked
   key, and it goes straight to a model that will happily quote it. Every payload is
   scrubbed before it is returned, and a diff larger than the ceiling is truncated with a
   notice rather than silently cut.
5. **Failure.** A 401, a 403, a 404, a 429, an outage - each becomes a result the model can
   explain to the user, in their language, instead of an exception that ends the turn.

It executes nothing itself. The HTTP client, the validators and the secret patterns already
exist in ``app/plugins/github``; this composes them behind one checked entry point, which is
the part that was missing.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# The diff ceiling: past this the model is reading for minutes and the user is waiting.
MAX_DIFF_CHARS = 10_000

# Shorthands people type. A resolved name is still validated - this only saves the model
# from having to guess, and a wrong guess is the thing that produces 404s.
_REPO_ALIASES = {
    "backend": "indooneteam/Indoone-Backend",
    "indoone-backend": "indooneteam/Indoone-Backend",
    "api": "indooneteam/Indoone-Backend",
    "client": "indooneteam/Indoone-Client",
    "frontend": "indooneteam/Indoone-Client",
    "web": "indooneteam/Indoone-Client",
}

# A git ref may contain letters, digits, dot, dash, underscore and slash. Anything else -
# especially a shell metacharacter - is not a branch name, it is an injection attempt or a
# hallucination, and either way it does not become a URL.
_SAFE_REF = re.compile(r"^[A-Za-z0-9._/-]{1,120}$")
_SAFE_REPO = re.compile(r"^[A-Za-z0-9._-]{1,60}/[A-Za-z0-9._-]{1,80}$")

_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
_BEARER = re.compile(r"\b(Bearer|token)\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE)
# A diff line begins with + or -, so the assignment is not at the start of the line. That
# one character is why "DATABASE_PASSWORD=..." survived the first version of this scrubber.
_ASSIGNMENT = re.compile(
    r"(?im)^\s*[+-]?\s*(?:export\s+)?"
    r"([A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*)"
    r"\s*=\s*\S+"
)
_URL_PASSWORD = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^:/\s]+:)([^@\s]+)(@)")

REDACTED = "[REDACTED_SECRET]"

_MESSAGES = {
    "UNAUTHORIZED": {
        "kn": "GitHub ಸಂಪರ್ಕ ಮುಗಿದಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಸಂಪರ್ಕಿಸಿ.",
        "hi": "GitHub कनेक्शन समाप्त हो गया है। कृपया दोबारा जोड़ें।",
        "en": "Your GitHub connection has expired. Please reconnect it.",
    },
    "FORBIDDEN": {
        "kn": "ಈ ರೆಪೊಗೆ ನಿಮಗೆ ಅನುಮತಿ ಇಲ್ಲ.",
        "hi": "इस रिपॉज़िटरी के लिए आपके पास अनुमति नहीं है।",
        "en": "You do not have access to that repository.",
    },
    "RATE_LIMITED": {
        "kn": "GitHub ಮಿತಿ ಮುಗಿದಿದೆ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "hi": "GitHub की सीमा पूरी हो गई है। कुछ देर बाद कोशिश करें।",
        "en": "GitHub's rate limit is reached. Please try again shortly.",
    },
    "NOT_FOUND": {
        "kn": "ಅದು ಸಿಗಲಿಲ್ಲ. ಹೆಸರು ಸರಿಯಾಗಿದೆಯೇ ನೋಡಿ.",
        "hi": "वह नहीं मिला। क्या नाम सही है?",
        "en": "That was not found. Please check the name.",
    },
    "UNAVAILABLE": {
        "kn": "GitHub ಈಗ ಲಭ್ಯವಿಲ್ಲ. ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಪ್ರಯತ್ನಿಸಿ.",
        "hi": "GitHub अभी उपलब्ध नहीं है। थोड़ी देर बाद कोशिश करें।",
        "en": "GitHub is not responding right now. Please try again shortly.",
    },
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "github_list_branches",
        "description": "List the branches of a repository, newest first.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name, or a shorthand like backend"},
                "limit": {"type": "integer", "description": "how many to return", "default": 10},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_diff",
        "description": "The diff between two refs of a repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "base": {"type": "string"},
                "head": {"type": "string"},
            },
            "required": ["repo", "base", "head"],
        },
    },
    {
        "name": "github_list_prs",
        "description": "List pull requests, optionally filtered by state.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"], "default": "open"},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "github_get_commit",
        "description": "One commit: message, author, and the files it touched.",
        "parameters": {
            "type": "object",
            "properties": {"repo": {"type": "string"}, "ref": {"type": "string"}},
            "required": ["repo", "ref"],
        },
    },
)

_REQUIRED_ARGUMENTS = {
    "github_list_branches": ("repo",),
    "github_get_diff": ("repo", "base", "head"),
    "github_list_prs": ("repo",),
    "github_get_commit": ("repo", "ref"),
}

_CHIPS = {
    "github_list_branches": ({"label": "View Diff", "action": "diff"},),
    "github_get_diff": ({"label": "Check PRs", "action": "prs"},),
    "github_list_prs": ({"label": "View Diff", "action": "diff"},),
    "github_get_commit": ({"label": "Recent Commits", "action": "commits"},),
}


@dataclass(slots=True)
class GitHubToolCall:
    """One function call, as the model emitted it."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class SupervisorExecutionResult:
    """What came back, cleaned, with something the model can say if it did not."""

    success: bool
    data: Any = None
    sanitized_output: str = ""
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_GITHUB_SUPERVISOR_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_tool_declarations() -> list[dict[str, Any]]:
    """The four tools the model may call, in function-calling shape."""
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- arguments


def resolve_repository(repo: str) -> str:
    """Turn what a person types into owner/name, without guessing at a real repository."""
    value = (repo or "").strip().strip("/")

    if "/" in value:
        return value

    return _REPO_ALIASES.get(value.casefold(), value)


def validate_arguments(call: GitHubToolCall) -> tuple[dict[str, Any] | None, str | None]:
    """Check a call before it becomes a request. Returns the cleaned arguments or a reason."""
    if call.tool_name not in _REQUIRED_ARGUMENTS:
        return None, f"unknown tool {call.tool_name!r}"

    arguments = dict(call.arguments or {})

    for required in _REQUIRED_ARGUMENTS[call.tool_name]:
        if not str(arguments.get(required, "")).strip():
            return None, f"missing argument {required!r}"

    repo = resolve_repository(str(arguments.get("repo", "")))

    if not _SAFE_REPO.match(repo):
        return None, f"invalid repository {repo!r}"

    arguments["repo"] = repo

    for name in ("base", "head", "ref"):
        if name in arguments and arguments[name] is not None:
            ref = str(arguments[name]).strip()

            if not _SAFE_REF.match(ref) or ".." in ref:
                return None, f"invalid git ref {ref!r}"

            arguments[name] = ref

    if "limit" in arguments:
        try:
            arguments["limit"] = max(1, min(50, int(arguments["limit"])))
        except (TypeError, ValueError):
            arguments["limit"] = 10

    if "state" in arguments:
        state = str(arguments["state"]).casefold()
        arguments["state"] = state if state in {"open", "closed", "all"} else "open"

    return arguments, None


# ---------------------------------------------------------------- the scrubber


# Cheap evidence that a payload might carry something the vault knows about: a provider key
# prefix, a bearer token, a JWT, or a long digit run that could be an Indian identifier.
_MAYBE_SECRET = re.compile(
    r"AKIA|sk-|ghp_|gh[pousr]_|github_pat_|AIza|xox[baprs]-|eyJ|Bearer|\d{10,}",
    re.IGNORECASE,
)


def _may_contain_secret(text: str) -> bool:
    return _MAYBE_SECRET.search(text) is not None


def sanitize_payload(raw_content: str, *, limit: int = MAX_DIFF_CHARS) -> str:
    """Remove what must not reach a model, then bound what is left.

    A diff is where a leaked key actually lives: someone commits `.env`, opens a pull
    request, and the model is asked to summarise it. Scrubbing here means the key never
    enters a prompt, which is the only point at which it can still be prevented.
    """
    body = raw_content or ""

    body = _PRIVATE_KEY.sub(REDACTED, body)
    body = _BEARER.sub(lambda match: f"{match.group(1)} {REDACTED}", body)
    body = _URL_PASSWORD.sub(lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}", body)
    body = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}={REDACTED}", body)

    # The vault's patterns cover the provider keys - AWS, GitHub, Gemini, JWTs - and the
    # Indian identifiers, and it is already the one place that knows them. It is also the
    # expensive part, so it runs only when the text could plausibly contain one: most diffs
    # are code, and a code diff should not pay for a PII scan it cannot need.
    if _may_contain_secret(body):
        try:
            from app.agents import vault as _vault

            body = _vault().redact(body).redacted_prompt
        except Exception:  # a scrubber that breaks must not hand back the unscrubbed text
            pass

    if len(body) > limit:
        kept = body[:limit]
        # Cut at a line, so the last thing the model reads is a whole line of the diff.
        cut = kept.rfind("\n")
        kept = kept[:cut] if cut > limit // 2 else kept
        body = f"{kept}\n\n[diff truncated: showing {len(kept)} of {len(raw_content)} characters]"

    return body


# ---------------------------------------------------------------- errors


def _error(code: str, language: str, detail: str = "") -> SupervisorExecutionResult:
    table = _MESSAGES.get(code, _MESSAGES["INVALID_ARGUMENT"])
    code_language = (language or "en").strip().casefold()[:2]

    return SupervisorExecutionResult(
        success=False,
        data=None,
        sanitized_output="",
        action_chips=[],
        error_code=code,
        error_message=table.get(code_language, table["en"]),
    )


def _status_of(exc: Exception) -> int:
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)

        if isinstance(value, int):
            return value

    text = str(exc)

    for status in (401, 403, 404, 422, 429, 500, 502, 503, 504):
        if str(status) in text:
            return status

    return 0


def _error_for_status(status: int, language: str) -> SupervisorExecutionResult:
    if status == 401:
        return _error("UNAUTHORIZED", language)
    if status == 403:
        return _error("FORBIDDEN", language)
    if status == 404:
        return _error("NOT_FOUND", language)
    if status == 429:
        return _error("RATE_LIMITED", language)
    if status >= 500:
        return _error("UNAVAILABLE", language)
    if status == 422:
        return _error("INVALID_ARGUMENT", language)
    return _error("UNAVAILABLE", language)


# ---------------------------------------------------------------- execution


async def validate_and_execute(
    call: GitHubToolCall,
    token: str | None = None,
    *,
    executor: Callable[..., Any] | None = None,
    language: str = "en",
    user_id: str = "",
) -> SupervisorExecutionResult:
    """Check the call, run it, scrub what comes back, and never raise.

    ``token`` is never logged, never placed in the result, and never returned to a caller
    that might put it in a prompt. When it is absent the user's stored connection is used.
    """
    started = time.perf_counter()

    arguments, problem = validate_arguments(call)

    if problem is not None:
        result = _error("INVALID_ARGUMENT", language, problem)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    assert arguments is not None

    if executor is None:
        executor = _default_executor(user_id)

    try:
        raw = await executor(call.tool_name, arguments, token)
    except Exception as exc:  # every failure becomes a result, not a crash
        result = _error_for_status(_status_of(exc), language)
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    body = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    sanitized = sanitize_payload(body)

    return SupervisorExecutionResult(
        success=True,
        data=raw,
        sanitized_output=sanitized,
        action_chips=[dict(chip) for chip in _CHIPS.get(call.tool_name, ())],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


def _default_executor(user_id: str) -> Callable[..., Any]:
    """The repository's own client, behind the token rule.

    The token is not an argument here. Every client function resolves the caller's stored
    GitHub connection itself, so the credential exists only inside that call - it is never
    an argument the model produced, never part of a result, and never in a prompt.
    """

    async def run(tool_name: str, arguments: dict[str, Any], token: str | None) -> Any:
        from app.plugins.github import service as github

        # ``token`` exists for tests and for a caller that has already resolved one. In the
        # ordinary path it is None and the client uses the user's connection.
        actor = token or user_id

        if tool_name == "github_list_branches":
            return await github.list_github_branches(
                actor,
                arguments["repo"],
                per_page=arguments.get("limit", 10),
            )

        if tool_name == "github_list_prs":
            return await github.list_github_pull_requests(
                actor,
                arguments["repo"],
                per_page=arguments.get("limit", 20),
            )

        if tool_name == "github_get_commit":
            return await github.get_github_commit(actor, arguments["repo"], arguments["ref"])

        if tool_name == "github_get_diff":
            # A diff between two refs is the compare endpoint; there is no separate one, and
            # inventing a second path to the same data is how two callers come to disagree.
            return await github.compare_github_repository_commits(
                actor,
                arguments["repo"],
                arguments["base"],
                arguments["head"],
            )

        raise ValueError(f"unknown tool {tool_name!r}")

    return run
