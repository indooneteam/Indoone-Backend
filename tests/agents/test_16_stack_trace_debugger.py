"""Agent 16 - the traceback debugger.

The property: the person gets the line they wrote, not the forty frames the framework wrote
on the way to it - and the password in the connection error never reaches a model.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

debugger = load("16_stack_trace_debugger")

PYTHON_STACK = '''Traceback (most recent call last):
  File "/home/ubuntu/Indoone-Backend/.venv/lib/python3.12/site-packages/uvicorn/protocols.py", line 42, in run
    await app(scope)
  File "/home/ubuntu/Indoone-Backend/.venv/lib/python3.12/site-packages/starlette/routing.py", line 71, in handle
    response = await func(request)
  File "/home/ubuntu/Indoone-Backend/app/api/chat/chat.py", line 88, in handle_chat
    return total / count
ZeroDivisionError: division by zero'''

NODE_STACK = '''TypeError: x is not a function
    at handler (/home/dev/app/src/index.js:12:5)
    at /home/dev/app/node_modules/express/lib/router.js:3:1'''


# ---------------------------------------------------------------- 1. python


def test_a_python_traceback_yields_the_type_the_message_and_the_line() -> None:
    report = debugger.diagnose_traceback(PYTHON_STACK)

    assert report.is_traceback is True
    assert report.runtime == "python"
    assert report.exception_type == "ZeroDivisionError"
    assert report.exception_message == "division by zero"
    assert report.culprit_frame is not None
    assert report.culprit_frame.file.endswith("app/api/chat/chat.py")
    assert report.culprit_frame.line == 88
    assert report.culprit_frame.func == "handle_chat"
    assert report.culprit_frame.code == "return total / count"


def test_framework_frames_are_marked_and_dropped() -> None:
    report = debugger.diagnose_traceback(PYTHON_STACK)

    assert len(report.app_frames) == 1
    assert all(frame.is_app_frame for frame in report.app_frames)
    assert "framework frames hidden" in report.sanitized_snippet


def test_the_culprit_is_the_innermost_application_frame() -> None:
    stack = '''Traceback (most recent call last):
  File "/srv/app/main.py", line 10, in outer
    inner()
  File "/srv/app/lib/util.py", line 22, in inner
    raise ValueError("bad")
  File "/usr/lib/python3.12/json/decoder.py", line 5, in decode
    pass
ValueError: bad'''

    report = debugger.diagnose_traceback(stack)

    assert report.culprit_frame is not None
    assert report.culprit_frame.file == "srv/app/lib/util.py"
    assert report.culprit_frame.line == 22


# ---------------------------------------------------------------- 2. other runtimes


def test_a_node_stack_is_identified_as_node() -> None:
    report = debugger.diagnose_traceback(NODE_STACK)

    assert report.is_traceback is True
    assert report.runtime == "node"
    assert report.exception_type == "TypeError"
    assert report.culprit_frame is not None
    assert report.culprit_frame.file.endswith("src/index.js")
    assert report.culprit_frame.line == 12


def test_a_node_frame_in_node_modules_is_noise() -> None:
    report = debugger.diagnose_traceback(NODE_STACK)

    assert len(report.app_frames) == 1


def test_plain_text_is_not_a_traceback() -> None:
    report = debugger.diagnose_traceback("hello, can you help me with my invoice?")

    assert report.is_traceback is False
    assert report.culprit_frame is None


def test_an_empty_payload_is_not_a_traceback() -> None:
    assert debugger.diagnose_traceback("").is_traceback is False
    assert debugger.is_error_payload("   ") is False


# ---------------------------------------------------------------- 3. the scrubber


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("could not connect: postgresql://admin:s3cretpass@db:5432/indoone", "s3cretpass"),
        ("mysql://root:hunter2@localhost/app failed", "hunter2"),
        ("redis://default:topsecret@cache:6379/0 refused", "topsecret"),
        ("mongodb+srv://user:pa55word@cluster/db timeout", "pa55word"),
    ],
)
def test_a_password_in_a_connection_error_is_scrubbed(text: str, secret: str) -> None:
    cleaned = debugger.sanitize_error_text(text)

    assert secret not in cleaned
    assert "[REDACTED_SECRET]" in cleaned


def test_a_bearer_token_is_scrubbed() -> None:
    cleaned = debugger.sanitize_error_text("401 from api: Authorization: Bearer abcdefghijklmnopqrstuvwxyz")

    assert "abcdefghijklmnopqrstuvwxyz" not in cleaned


def test_a_home_directory_is_normalised() -> None:
    cleaned = debugger.sanitize_error_text('File "/home/ubuntu/Indoone-Backend/app/main.py", line 3')

    assert "/home/ubuntu" not in cleaned
    assert "[ROOT]" in cleaned


def test_a_password_assignment_is_scrubbed() -> None:
    cleaned = debugger.sanitize_error_text("settings error: password=hunter2hunter2 rejected")

    assert "hunter2" not in cleaned


# ---------------------------------------------------------------- 4. the free fixes


def test_a_missing_module_is_answered_without_a_model() -> None:
    report = debugger.diagnose_traceback("ModuleNotFoundError: No module named 'requests'")

    assert report.deterministic_fix == "Run: pip install requests"
    assert "pip install requests" in report.sanitized_snippet


def test_a_dotted_missing_module_installs_the_top_level_package() -> None:
    report = debugger.diagnose_traceback("ModuleNotFoundError: No module named 'google.cloud.storage'")

    assert report.deterministic_fix == "Run: pip install google"


def test_an_indentation_error_names_the_line() -> None:
    report = debugger.diagnose_traceback(PYTHON_STACK.replace("ZeroDivisionError: division by zero", "IndentationError: unexpected indent"))

    assert report.deterministic_fix is not None
    assert "indentation" in report.deterministic_fix.lower()
    assert "88" in report.deterministic_fix


def test_a_division_by_zero_names_the_line_and_the_file() -> None:
    report = debugger.diagnose_traceback(PYTHON_STACK)

    assert report.deterministic_fix is not None
    assert "88" in report.deterministic_fix
    assert "chat.py" in report.deterministic_fix


def test_a_complex_error_gets_no_invented_fix() -> None:
    """When there is no single answer, saying nothing is the honest answer."""
    report = debugger.diagnose_traceback(PYTHON_STACK.replace("ZeroDivisionError: division by zero", "KeyError: 'user_id'"))

    assert report.deterministic_fix is None


# ---------------------------------------------------------------- 5. contract and speed


def test_the_report_carries_the_chips() -> None:
    report = debugger.diagnose_traceback(PYTHON_STACK)

    actions = [chip["action"] for chip in report.action_chips]
    assert actions == ["test_in_sandbox", "show_code_line"]


def test_the_tool_declaration_is_shaped_and_immutable() -> None:
    declarations = debugger.get_tool_declarations()
    declarations[0]["name"] = "something_else"

    assert debugger.get_tool_declarations()[0]["name"] == "diagnose_error_log"


def test_the_tool_runs_end_to_end() -> None:
    result = debugger.validate_and_execute(
        debugger.DebuggerToolCall("diagnose_error_log", {"log": PYTHON_STACK})
    )

    assert result.success is True
    assert result.report is not None
    assert result.report.exception_type == "ZeroDivisionError"


def test_an_empty_log_is_refused_cleanly() -> None:
    result = debugger.validate_and_execute(debugger.DebuggerToolCall("diagnose_error_log", {"log": ""}))

    assert result.success is False
    assert result.error_code == "INVALID_ARGUMENT"


def test_parsing_a_traceback_stays_under_its_budget() -> None:
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        debugger.diagnose_traceback(PYTHON_STACK)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"
