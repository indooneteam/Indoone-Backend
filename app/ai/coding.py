from __future__ import annotations

import ast
import json
from dataclasses import dataclass

MAX_CODE_LENGTH = 12_000
SUPPORTED_LANGUAGES = frozenset({"python", "javascript", "typescript", "java", "kotlin", "c", "cpp", "go", "rust", "bash", "sql"})


@dataclass(frozen=True)
class CodeIssue:
    severity: str
    message: str
    line: int = 0
    column: int = 0


def _python_issues(code: str) -> list[CodeIssue]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return [
            CodeIssue(
                severity="error",
                message=exc.msg,
                line=exc.lineno or 0,
                column=exc.offset or 0,
            )
        ]
    return []


def _delimiter_issues(code: str) -> list[CodeIssue]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = set(pairs.values())
    stack: list[tuple[str, int, int]] = []
    issues: list[CodeIssue] = []
    for line_no, line in enumerate(code.splitlines(), start=1):
        for column, char in enumerate(line, start=1):
            if char in pairs:
                stack.append((char, line_no, column))
            elif char in closing:
                if not stack or pairs[stack[-1][0]] != char:
                    issues.append(CodeIssue("error", f"unexpected '{char}'", line_no, column))
                else:
                    stack.pop()
    for opening, line_no, column in reversed(stack):
        issues.append(CodeIssue("error", f"unclosed '{opening}'", line_no, column))
    return issues[:20]


def analyze_code(language: str, code: str) -> dict[str, object]:
    language = language.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"code exceeds {MAX_CODE_LENGTH} characters")

    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    issues = _python_issues(normalized) if language == "python" else _delimiter_issues(normalized)
    return {
        "language": language,
        "characters": len(normalized),
        "lines": len(normalized.splitlines()) if normalized else 0,
        "valid": not issues,
        "issues": [issue.__dict__ for issue in issues],
    }


def code_analysis_tool(payload: str) -> str:
    data = json.loads(payload)
    language = str(data.get("language", ""))
    code = data.get("code", "")
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    return json.dumps(analyze_code(language, code), ensure_ascii=False, sort_keys=True)
