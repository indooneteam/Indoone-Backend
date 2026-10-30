"""Agent 16 - the debugger: find the bug, not the framework.

A two-hundred-line traceback pasted into a chat is mostly noise. The framework's own frames
bury the one line the person actually wrote, the database URL carries a password, and the
whole thing costs a thousand tokens to say "you forgot to import requests".

So this reads the traceback before the model does:

* **the culprit** is the innermost frame that belongs to application code - the last frame
  that is not ``site-packages``, ``.venv`` or ``node_modules``;
* **the noise** is dropped, not summarised: forty framework frames become a count;
* **the secrets** go before anything else sees them, because a connection error is exactly
  where a password appears;
* **the obvious fixes** are answered without a model at all - a missing module, an
  indentation error and a division by zero do not need one, and asking one costs money and
  can produce a wrong answer to a question with a single right one.

It parses; it does not run anything and does not fetch anything.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# Frames from these are the framework's, not the person's.
_NOISE_MARKERS = (
    "site-packages", "/.venv/", "/venv/", "/lib/python3", "/lib/python2", "/python3.",
    "node_modules", "/usr/lib/python", "<frozen", "dist-packages", "/usr/local/lib/python",
)

_PY_TRACEBACK = re.compile(r"Traceback \(most recent call last\)")
_PY_FRAME = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>.+?)\s*$')
_PY_ERROR = re.compile(r"^(?P<type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning|Interrupt|Exit|Fault|Iteration))\s*:?\s*(?P<message>.*)$")
# MULTILINE, because a stack is a document and the error line is its first: without it the
# pattern only matched when the error was the entire string, which a real stack never is.
_NODE_ERROR = re.compile(
    r"^(?P<type>[A-Za-z_][A-Za-z0-9_]*Error)\s*:\s*(?P<message>.*)$", re.MULTILINE
)
_NODE_FRAME = re.compile(r"^\s*at\s+(?:(?P<func>[^\s(]+)\s+)?\(?(?P<file>[^():]+):(?P<line>\d+):(?P<col>\d+)\)?\s*$")
_GO_PANIC = re.compile(r"^(?:panic:|goroutine \d+)", re.MULTILINE)
_SQL_ERROR = re.compile(r"\b(?:psycopg2|sqlalchemy|OperationalError|IntegrityError|ProgrammingError|SQLSTATE)\b", re.IGNORECASE)

# A password in a URI, a bearer token, and a home directory that names a person.
_URI_PASSWORD = re.compile(r"\b((?:postgres(?:ql)?|mysql|mariadb|redis|mongodb(?:\+srv)?|amqp|mssql)://[^:\s/@]+:)([^@\s]+)(@)", re.IGNORECASE)
_BEARER = re.compile(r"\b(Bearer|token)\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE)
_HOME_PATH = re.compile(r"(?P<prefix>/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\\\Users\\\\[^\\\s]+)")
_KEY_VALUE = re.compile(r"(?i)\b(password|passwd|pwd|secret|api_key|token)\s*=\s*[^\s,;)]+")

REDACTED = "[REDACTED_SECRET]"
ROOT = "[ROOT]"

_MESSAGES = {
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
}


@dataclass(slots=True)
class StackFrame:
    file: str
    line: int
    func: str
    code: str = ""
    is_app_frame: bool = True


@dataclass(slots=True)
class DiagnosticReport:
    is_traceback: bool
    runtime: str = "generic"
    exception_type: str = ""
    exception_message: str = ""
    culprit_frame: StackFrame | None = None
    app_frames: list[StackFrame] = field(default_factory=list)
    sanitized_snippet: str = ""
    deterministic_fix: str | None = None
    action_chips: list[dict[str, str]] = field(default_factory=list)
    telemetry_micros: float = 0.0


def enabled() -> bool:
    import os

    return os.getenv("INDOONE_STACK_TRACE_DEBUGGER_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


# ---------------------------------------------------------------- the scrubber


def sanitize_error_text(text: str) -> str:
    """Remove the credentials and the home directories, keep the error."""
    body = text or ""

    body = _URI_PASSWORD.sub(lambda match: f"{match.group(1)}{REDACTED}{match.group(3)}", body)
    body = _BEARER.sub(lambda match: f"{match.group(1)} {REDACTED}", body)
    body = _KEY_VALUE.sub(lambda match: f"{match.group(1)}={REDACTED}", body)
    body = _HOME_PATH.sub(ROOT, body)

    # The vault knows the provider keys and the Indian identifiers; it runs only when the
    # text could plausibly hold one, because most tracebacks are about a NoneType.
    if re.search(r"AKIA|sk-|ghp_|gh[pousr]_|AIza|eyJ|\d{10,}", body):
        try:
            from app.agents import vault as _vault

            body = _vault().redact(body).redacted_prompt
        except Exception:
            pass

    return body


# ---------------------------------------------------------------- detection


def is_error_payload(text: str) -> bool:
    """Whether this looks like something worth diagnosing at all."""
    body = text or ""

    if not body.strip():
        return False

    return bool(
        _PY_TRACEBACK.search(body)
        or _NODE_ERROR.search(body)
        or _GO_PANIC.search(body)
        or _SQL_ERROR.search(body)
        or re.search(r"^\s*File \"[^\"]+\", line \d+", body, re.MULTILINE)
    )


def _is_noise(path: str) -> bool:
    lowered = path.replace("\\", "/").casefold()
    return any(marker.casefold() in lowered for marker in _NOISE_MARKERS)


def _shorten(path: str) -> str:
    """A frame's file, relative to the project root when it is under a home directory."""
    cleaned = _HOME_PATH.sub("", path).lstrip("/\\")
    return cleaned or path


# ---------------------------------------------------------------- parsing


def _python_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    frames: list[StackFrame] = []
    exception_type = ""
    exception_message = ""
    pending: StackFrame | None = None

    for line in lines:
        match = _PY_FRAME.match(line)

        if match:
            if pending is not None:
                frames.append(pending)

            path = match.group("file")
            pending = StackFrame(
                file=_shorten(path),
                line=int(match.group("line")),
                func=match.group("func"),
                is_app_frame=not _is_noise(path),
            )
            continue

        if pending is not None and line.strip() and line.startswith("    ") and not pending.code:
            pending.code = line.strip()
            continue

        error = _PY_ERROR.match(line.strip())

        if error and not line.startswith(" "):
            if pending is not None:
                frames.append(pending)
                pending = None

            exception_type = error.group("type")
            exception_message = error.group("message").strip()

    if pending is not None:
        frames.append(pending)

    return frames, exception_type, exception_message


def _node_frames(lines: list[str]) -> tuple[list[StackFrame], str, str]:
    frames: list[StackFrame] = []
    exception_type = ""
    exception_message = ""

    for line in lines:
        error = _NODE_ERROR.match(line.strip())

        if error and not exception_type:
            exception_type = error.group("type")
            exception_message = error.group("message").strip()
            continue

        frame = _NODE_FRAME.match(line)

        if frame:
            path = frame.group("file")
            frames.append(
                StackFrame(
                    file=_shorten(path),
                    line=int(frame.group("line")),
                    func=(frame.group("func") or "<anonymous>").strip(),
                    is_app_frame=not _is_noise(path),
                )
            )

    return frames, exception_type, exception_message


def _deterministic_fix(
    exception_type: str,
    exception_message: str,
    culprit: StackFrame | None,
    raw: str,
) -> str | None:
    """The answer, when there is only one answer.

    A missing module has one fix. Asking a model for it costs money and can produce a
    different one, which is worse than the sentence it replaces.
    """
    kind = (exception_type or "").rsplit(".", 1)[-1]

    if kind == "ModuleNotFoundError":
        match = re.search(r"No module named '([^']+)'", exception_message)
        module = match.group(1) if match else ""
        package = module.split(".")[0]

        if package:
            return f"Run: pip install {package}"

        return "A module is missing; install the package that provides it."

    if kind in {"IndentationError", "TabError"}:
        where = f"line {culprit.line}" if culprit else "the reported line"
        return f"Fix the indentation at {where} - Python blocks must line up."

    if kind == "SyntaxError":
        match = re.search(r"\(([^,]+), line (\d+)\)", raw)
        where = f"{match.group(1)} line {match.group(2)}" if match else "the reported line"
        return f"Fix the syntax at {where}."

    if kind == "ZeroDivisionError":
        where = f"line {culprit.line} of {culprit.file}" if culprit else "the reported line"
        return f"Guard the division at {where} against a zero denominator."

    return None


def diagnose_traceback(raw_log: str) -> DiagnosticReport:
    """Read a traceback and report the bug, the line, and the fix if there is one."""
    started = time.perf_counter()
    raw = raw_log or ""
    lines = raw.splitlines()

    if not is_error_payload(raw):
        return DiagnosticReport(
            is_traceback=False,
            sanitized_snippet=sanitize_error_text(raw.strip())[:400],
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    if _PY_TRACEBACK.search(raw) or re.search(r'^\s*File "[^"]+", line \d+', raw, re.MULTILINE):
        runtime = "python"
        frames, exception_type, exception_message = _python_frames(lines)
    elif _NODE_ERROR.search(raw):
        runtime = "node"
        frames, exception_type, exception_message = _node_frames(lines)
    elif _GO_PANIC.search(raw):
        runtime = "go"
        frames, exception_type, exception_message = [], "", lines[0] if lines else ""
    else:
        runtime = "sql"
        frames, exception_type, exception_message = [], "", ""

    if not exception_type:
        # Fall back to the last non-empty line, which is where every runtime puts it.
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                if ":" in stripped:
                    head, _, tail = stripped.partition(":")
                    exception_type, exception_message = head.strip(), tail.strip()
                else:
                    exception_message = stripped
                break

    app_frames = [frame for frame in frames if frame.is_app_frame]
    culprit = app_frames[-1] if app_frames else None

    snippet = sanitize_error_text(raw)
    fix = _deterministic_fix(exception_type, exception_message, culprit, raw)

    location = ""
    if culprit is not None:
        location = f"{culprit.file}:{culprit.line} in {culprit.func}"
        if culprit.code:
            location += f" -> {culprit.code}"

    summary_parts = [
        f"{exception_type}: {exception_message}".strip(": ").strip(),
        f"at {location}" if location else "",
        f"({len(frames) - len(app_frames)} framework frames hidden)" if frames and len(frames) != len(app_frames) else "",
    ]
    summary = "\n".join(part for part in summary_parts if part)

    if fix:
        summary += f"\n{fix}"

    return DiagnosticReport(
        is_traceback=True,
        runtime=runtime,
        exception_type=exception_type,
        exception_message=exception_message,
        culprit_frame=culprit,
        app_frames=app_frames,
        sanitized_snippet=summary,
        deterministic_fix=fix,
        action_chips=[
            {"label": "Run in Sandbox", "action": "test_in_sandbox"},
            {"label": "Show Offending Line", "action": "show_code_line"},
        ],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )


# ---------------------------------------------------------------- the tool


@dataclass(slots=True)
class DebuggerToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class DebuggerResult:
    success: bool
    report: DiagnosticReport | None = None
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def get_tool_declarations() -> list[dict[str, Any]]:
    declaration = {
        "name": "diagnose_error_log",
        "description": "Read a stack trace or error log and report the bug, its line and a fix.",
        "parameters": {
            "type": "object",
            "properties": {"log": {"type": "string"}},
            "required": ["log"],
        },
    }
    return [json.loads(json.dumps(declaration))]


def validate_and_execute(call: DebuggerToolCall, *, language: str = "en") -> DebuggerResult:
    started = time.perf_counter()

    if call.tool_name != "diagnose_error_log":
        table = _MESSAGES["INVALID_ARGUMENT"]
        key = (language or "en").strip().casefold()[:2]
        return DebuggerResult(
            success=False,
            error_code="INVALID_ARGUMENT",
            error_message=table.get(key, table["en"]),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    log = str((call.arguments or {}).get("log") or "")

    if not log.strip():
        table = _MESSAGES["INVALID_ARGUMENT"]
        key = (language or "en").strip().casefold()[:2]
        return DebuggerResult(
            success=False,
            error_code="INVALID_ARGUMENT",
            error_message=table.get(key, table["en"]),
            telemetry_micros=(time.perf_counter() - started) * 1_000_000,
        )

    report = diagnose_traceback(log)

    return DebuggerResult(
        success=True,
        report=report,
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
