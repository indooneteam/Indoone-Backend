"""Agent 15 - the code sandbox.

Two properties, and the second one is the reason this agent exists:

1. useful code runs - arithmetic, algorithms, regex, json;
2. code that would touch the machine does not, and a runaway loop is stopped by a process
   boundary rather than by a polite request.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

sandbox = load("15_code_sandbox_runner")


def run(code: str, timeout_seconds: float | None = None, **kwargs):
    arguments: dict[str, object] = {"code": code}
    if timeout_seconds is not None:
        arguments["timeout_seconds"] = timeout_seconds
    return sandbox.validate_and_execute(sandbox.SandboxToolCall("execute_python_snippet", arguments), **kwargs)


# ---------------------------------------------------------------- 1. declarations


def test_the_declaration_is_function_calling_shaped() -> None:
    declarations = sandbox.get_tool_declarations()

    assert [item["name"] for item in declarations] == ["execute_python_snippet"]
    assert declarations[0]["parameters"]["type"] == "object"
    assert declarations[0]["parameters"]["required"] == ["code"]


def test_a_caller_cannot_change_the_declaration() -> None:
    sandbox.get_tool_declarations()[0]["name"] = "execute_shell"

    assert sandbox.get_tool_declarations()[0]["name"] == "execute_python_snippet"


# ---------------------------------------------------------------- 2. the gate


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "from sys import *",
        "from os import system",
        "__import__('os').system('id')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd', 'r').read()",
        "eval('1 + 1')",
        "exec('x = 1')",
        "compile('1', '<s>', 'eval')",
        "getattr(object, '__subclasses__')",
        "x.__globals__",
        "globals()",
        "input()",
    ],
)
def test_a_dangerous_snippet_is_refused_before_it_runs(code: str) -> None:
    safe, reason = sandbox.audit_ast_safety(code)

    assert safe is False
    assert reason


def test_a_syntax_error_is_refused_cleanly() -> None:
    safe, reason = sandbox.audit_ast_safety("def broken(:\n    pass")

    assert safe is False
    assert "syntax error" in reason


def test_the_refusal_never_reaches_the_runner() -> None:
    result = run("import os\nos.system('rm -rf /')")

    assert result.success is False
    assert result.error_code == "BLOCKED_SECURITY_VIOLATION"
    assert result.stdout == ""


@pytest.mark.parametrize(
    "code",
    [
        "result = sum(range(100))",
        "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nresult = f(12)",
        "result = sorted([3, 1, 2])",
        "result = re.findall(r'[0-9]+', 'a1b22c333')",
        "result = json.dumps({'a': 1})",
        "result = math.factorial(10)",
        "result = statistics.mean([1, 2, 3])",
        "print('hello')",
    ],
)
def test_ordinary_code_is_allowed(code: str) -> None:
    safe, reason = sandbox.audit_ast_safety(code)

    assert safe is True, reason


# ---------------------------------------------------------------- 3. execution


def test_arithmetic_runs_and_returns_a_value() -> None:
    result = run("result = 6 * 7\nprint(result)")

    assert result.success is True
    assert result.stdout == "42"
    assert result.return_value == "42"


def test_fibonacci_runs() -> None:
    result = run("def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nprint(f(10))")

    assert result.success is True
    assert result.stdout == "55"


def test_a_regex_match_runs() -> None:
    result = run("result = re.findall(r'[0-9]+', 'a1b22c333')\nprint(result)")

    assert result.success is True
    assert "333" in result.stdout


def test_json_round_trips() -> None:
    result = run("result = json.loads('{\"a\": [1, 2]}')\nprint(result['a'][1])")

    assert result.success is True
    assert result.stdout == "2"


def test_an_error_comes_back_as_a_result() -> None:
    result = run("result = 1 / 0")

    assert result.success is False
    assert result.error_code == "EXECUTION_ERROR"
    assert "ZeroDivisionError" in result.stderr


# ---------------------------------------------------------------- 4. the boundary


def test_an_infinite_loop_is_stopped_by_the_process_boundary() -> None:
    started = time.monotonic()
    result = run("while True:\n    pass", timeout_seconds=1.0)
    elapsed = time.monotonic() - started

    assert result.success is False
    assert result.error_code == "TIMEOUT_EXCEEDED"
    assert elapsed < 6.0, f"took {elapsed:.1f}s"


def test_the_timeout_is_clamped_to_a_sane_maximum() -> None:
    """A caller cannot ask for an hour."""
    started = time.monotonic()
    result = run("while True:\n    pass", timeout_seconds=600)
    elapsed = time.monotonic() - started

    assert result.error_code == "TIMEOUT_EXCEEDED"
    assert elapsed < sandbox.MAX_TIMEOUT + 4


def test_output_is_clamped_with_a_notice() -> None:
    result = run("print('x' * 5000)")

    assert result.success is True
    assert len(result.stdout) < sandbox.MAX_OUTPUT_CHARS + 100
    assert "[Output truncated at 4000 characters]" in result.stdout


# ---------------------------------------------------------------- 5. speed


def test_the_ast_audit_costs_under_its_budget() -> None:
    code = "def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nresult = f(12)"
    fastest = float("inf")

    for _ in range(200):
        started = time.thread_time()
        sandbox.audit_ast_safety(code)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


# ---------------------------------------------------------------- 6. chips


def test_a_successful_run_offers_explain_and_run_again() -> None:
    result = run("result = 1 + 1")

    actions = [chip["action"] for chip in result.action_chips]
    assert actions == ["explain_snippet", "new_snippet"]


def test_a_blocked_run_offers_debug() -> None:
    result = run("import os")

    assert [chip["action"] for chip in result.action_chips] == ["debug_snippet"]
