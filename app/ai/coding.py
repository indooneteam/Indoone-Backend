from __future__ import annotations

import ast
import json
from dataclasses import dataclass

MAX_CODE_LENGTH = 12_000
SUPPORTED_LANGUAGES = frozenset({"python", "javascript", "typescript", "java", "kotlin", "c", "cpp", "go", "rust", "bash", "sql"})
SUPPORTED_TRANSFORMS = frozenset({"normalize", "strip_trailing_whitespace", "ensure_final_newline"})


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
        return [CodeIssue("error", exc.msg, exc.lineno or 0, exc.offset or 0)]
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


def explain_code(language: str, code: str) -> dict[str, object]:
    """Return a deterministic structural explanation without executing user code."""
    language = language.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"code exceeds {MAX_CODE_LENGTH} characters")
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    analysis = analyze_code(language, normalized)
    lines = normalized.splitlines()
    concepts: list[str] = []
    if language == "python":
        try:
            tree = ast.parse(normalized)
            counts = {
                "functions": sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)),
                "classes": sum(isinstance(node, ast.ClassDef) for node in ast.walk(tree)),
                "imports": sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)),
                "conditionals": sum(isinstance(node, ast.If) for node in ast.walk(tree)),
                "loops": sum(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree)),
            }
            concepts.extend(name for name, count in counts.items() if count)
        except SyntaxError:
            concepts.append("syntax-error")
    else:
        lowered = normalized.lower()
        if "import " in lowered or "#include" in lowered:
            concepts.append("imports/includes")
        if "function " in lowered or "func " in lowered or "def " in lowered:
            concepts.append("functions")
        if "class " in lowered:
            concepts.append("classes")
        if " if " in f" {lowered} ":
            concepts.append("conditionals")
        if " for " in f" {lowered} " or " while " in f" {lowered} ":
            concepts.append("loops")
    return {
        "language": language,
        "characters": analysis["characters"],
        "lines": analysis["lines"],
        "valid": analysis["valid"],
        "issues": analysis["issues"],
        "non_empty_lines": sum(bool(line.strip()) for line in lines),
        "concepts": concepts,
        "summary": f"{language} code with {len(lines)} lines and {len(concepts)} detected structural concepts.",
    }


def suggest_fixes(language: str, code: str) -> dict[str, object]:
    """Return deterministic, non-executing repair guidance for detected issues."""
    analysis = analyze_code(language, code)
    suggestions: list[dict[str, object]] = []
    for issue in analysis["issues"]:
        message = str(issue["message"])
        if "unclosed '" in message:
            opening = message.split("'")[1]
            closing = {"(": ")", "[": "]", "{": "}"}.get(opening, "")
            advice = f"Add the matching closing delimiter '{closing}' near line {issue['line']}." if closing else "Add the matching closing delimiter."
        elif "unexpected '" in message:
            advice = "Check the nearby delimiters and remove or replace the unexpected closing character."
        elif language == "python" and message:
            advice = "Review the indicated Python syntax around the reported line and column; fix the construct described by the parser."
        else:
            advice = "Review the reported issue at the indicated location before changing behavior."
        suggestions.append({"line": issue["line"], "column": issue["column"], "severity": issue["severity"], "issue": message, "suggestion": advice})
    if not suggestions:
        suggestions.append({"line": 0, "column": 0, "severity": "info", "issue": "none", "suggestion": "No static syntax or delimiter issue was detected."})
    return {"language": analysis["language"], "valid": analysis["valid"], "issue_count": len(analysis["issues"]), "suggestions": suggestions, "safe": True, "execution": "not_performed"}


def transform_code(language: str, code: str, operation: str = "normalize") -> dict[str, object]:
    """Apply deterministic text-only code transformations; never executes user code."""
    language = language.strip().lower()
    operation = operation.strip().lower()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported language: {language}")
    if operation not in SUPPORTED_TRANSFORMS:
        raise ValueError(f"unsupported transform: {operation}")
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"code exceeds {MAX_CODE_LENGTH} characters")

    transformed = code.replace("\r\n", "\n").replace("\r", "\n")
    if operation in {"normalize", "strip_trailing_whitespace"}:
        transformed = "\n".join(line.rstrip() for line in transformed.split("\n"))
    if operation in {"normalize", "ensure_final_newline"} and transformed and not transformed.endswith("\n"):
        transformed += "\n"
    return {
        "language": language,
        "operation": operation,
        "changed": transformed != code,
        "code": transformed,
        "safe": True,
        "execution": "not_performed",
    }


def code_analysis_tool(payload: str) -> str:
    data = json.loads(payload)
    language = str(data.get("language", ""))
    code = data.get("code", "")
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    return json.dumps(analyze_code(language, code), ensure_ascii=False, sort_keys=True)


def code_fix_suggestions_tool(payload: str) -> str:
    data = json.loads(payload)
    language = str(data.get("language", ""))
    code = data.get("code", "")
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    return json.dumps(suggest_fixes(language, code), ensure_ascii=False, sort_keys=True)


def code_transform_tool(payload: str) -> str:
    data = json.loads(payload)
    language = str(data.get("language", ""))
    code = data.get("code", "")
    operation = str(data.get("operation", "normalize"))
    if not isinstance(code, str):
        raise ValueError("code must be a string")
    return json.dumps(transform_code(language, code, operation), ensure_ascii=False, sort_keys=True)
