"""Agent 15 - the sandbox: run the arithmetic, not the machine.

A model doing arithmetic in its head gets it wrong often enough to matter, and a model asked
to run code that is handed a shell gets the whole server wrong. This is the boundary between
those two facts.

**The AST gate comes first, and it is a whitelist of shapes, not a blacklist of words.** A
blocklist of dangerous names is a puzzle to be solved - ``__builtins__['__import__']`` is not
``__import__``, and ``getattr(x, "__gl" + "obals__")`` is not an attribute called
``__globals__``. So: no imports at all (the useful modules are pre-injected), no attribute
whose name starts with an underscore, no call to a name that is not in the injected
namespace. That is checkable, and it is checked before anything is compiled.

**Execution happens in a separate process, not in this one.** This is the part that cannot be
compromised on: a restricted ``exec`` in the same interpreter is not a sandbox - a runaway
loop cannot be interrupted in a thread, and one ``__subclasses__`` walk reaches the
interpreter's own classes. The snippet runs in a child process with a CPU limit, an address
space limit, no writable file size, and a wall-clock kill.

The output comes back bounded, because a page of printed numbers is not an answer.
"""

from __future__ import annotations

import ast
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

MAX_OUTPUT_CHARS = 4_000
DEFAULT_TIMEOUT = 2.0
MAX_TIMEOUT = 5.0
MEMORY_LIMIT_BYTES = 64 * 1024 * 1024

# Names the snippet may call. Everything else - open, eval, exec, compile, input, vars,
# getattr, setattr, __import__ - is absent, and absence is easier to verify than prohibition.
_SAFE_BUILTINS = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "chr", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash", "hex",
    "int", "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next",
    "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted",
    "str", "sum", "tuple", "type", "zip", "print",
)

# Modules injected into the namespace, so a snippet that needs maths or a regex never needs
# an import statement - and "no imports at all" stays a rule with no exceptions.
_SAFE_MODULES = ("math", "re", "json", "datetime", "collections", "itertools", "statistics", "fractions", "decimal")

# Hoisted out of the audit: this tuple was rebuilt on every call, which is most of the
# budget for a check that is meant to be free.
_ALLOWED_NODES = (
        ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While,
        ast.If, ast.IfExp, ast.FunctionDef, ast.Return, ast.Pass, ast.Break, ast.Continue,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
        ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Attribute, ast.Name,
        ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice,
        ast.JoinedStr, ast.FormattedValue, ast.keyword, ast.arguments, ast.arg,
        ast.Load, ast.Store, ast.operator, ast.unaryop, ast.cmpop, ast.boolop, ast.expr_context,
    ast.Starred, ast.Import, ast.ImportFrom, ast.alias,
)

_MESSAGES = {
    "BLOCKED_SECURITY_VIOLATION": {
        "kn": "ಈ ಕೋಡ್ ಸುರಕ್ಷಿತವಲ್ಲ, ಹಾಗಾಗಿ ರನ್ ಮಾಡಿಲ್ಲ.",
        "hi": "यह कोड सुरक्षित नहीं है, इसलिए नहीं चलाया गया।",
        "en": "That code is not safe to run, so it was not executed.",
    },
    "TIMEOUT_EXCEEDED": {
        "kn": "ಕೋಡ್ ತುಂಬಾ ಸಮಯ ತೆಗೆದುಕೊಂಡಿತು, ನಿಲ್ಲಿಸಲಾಗಿದೆ.",
        "hi": "कोड बहुत समय ले रहा था, रोक दिया गया।",
        "en": "That code ran too long and was stopped.",
    },
    "EXECUTION_ERROR": {
        "kn": "ಕೋಡ್ ರನ್ ಆಗುವಾಗ ದೋಷ ಬಂತು.",
        "hi": "कोड चलाते समय त्रुटि हुई।",
        "en": "The code raised an error while running.",
    },
    "INVALID_ARGUMENT": {
        "kn": "ಆ ಮಾಹಿತಿ ಸರಿಯಾಗಿಲ್ಲ.",
        "hi": "वह जानकारी सही नहीं है।",
        "en": "That request was not valid.",
    },
}

_TOOL_DECLARATIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "execute_python_snippet",
        "description": "Run a short Python snippet for arithmetic, algorithms or data work.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {"type": "number", "default": DEFAULT_TIMEOUT},
            },
            "required": ["code"],
        },
    },
)


@dataclass(slots=True)
class SandboxToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


@dataclass(slots=True)
class SandboxExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_value: str | None = None
    action_chips: list[dict[str, str]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    telemetry_micros: float = 0.0


def enabled() -> bool:
    return os.getenv("INDOONE_CODE_SANDBOX_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off",
    }


def get_tool_declarations() -> list[dict[str, Any]]:
    return [json.loads(json.dumps(declaration)) for declaration in _TOOL_DECLARATIONS]


# ---------------------------------------------------------------- the gate


def _is_banned_name(name: str) -> bool:
    """Anything that reaches the interpreter's own machinery."""
    return name.startswith("_")


_CALLABLE_NAMES = frozenset(_SAFE_BUILTINS) | frozenset(_SAFE_MODULES)


def audit_ast_safety(code: str) -> tuple[bool, str | None]:
    """Whether this snippet may run at all.

    A whitelist of shapes: a statement that is not an expression, an assignment, a loop, a
    branch, a function definition or a comprehension is refused, and any name that is not in
    the injected namespace is refused. Refusing by shape is the only version of this that
    survives a determined model.
    """
    source = code or ""

    if not source.strip():
        return False, "empty snippet"

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc.msg}"

    # One walk, collected once: the tree is traversed for names first, then checked.
    nodes = list(ast.walk(tree))

    # Names the snippet may call: the injected builtins, the injected modules, and whatever
    # it defined itself. A call to anything else - open, eval, exec, compile, input - is
    # refused by absence rather than by a list of forbidden words.
    defined: set[str] = set()

    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)

    callable_names = _CALLABLE_NAMES | defined

    for node in nodes:
        # isinstance, not type(): ast.Add is an ast.operator, and the concrete class is
        # what the walk yields.
        if not isinstance(node, _ALLOWED_NODES):
            return False, f"{type(node).__name__} is not allowed"

        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return False, "imports are not allowed; maths, re and json are already available"

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                return False, f"attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Name):
            if node.id.startswith("_"):
                return False, f"name {node.id!r} is not allowed"
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                return False, f"function name {node.name!r} is not allowed"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in callable_names:
                return False, f"{node.func.id}() is not available here"

    return True, None


# ---------------------------------------------------------------- the runner

_RUNNER = r'''
import json, sys
sys.path.insert(0, ".")
payload = json.loads(sys.stdin.read())
namespace = {"__builtins__": payload["builtins"]}
for name in payload["modules"]:
    try:
        namespace[name] = __import__(name)
    except Exception:
        pass
code = payload["code"]
try:
    compiled = compile(code, "<snippet>", "exec")
    exec(compiled, namespace)
    value = namespace.get("result")
    if value is not None:
        print("\n[result] " + repr(value))
except SystemExit:
    pass
except BaseException as exc:
    print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
'''


def _limits(cpu_seconds: int = 3) -> None:
    """What the child may not do: burn CPU, take memory, or write a file."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ValueError, OSError):
        pass


def run_isolated(code: str, timeout_sec: float = DEFAULT_TIMEOUT) -> tuple[str, str, str | None]:
    """Run a snippet in a child process and bring back what it printed.

    A separate process is the only way to stop a runaway loop: a thread cannot be
    interrupted, and a signal handler in a threaded server reaches the wrong thread.
    """
    timeout = max(0.1, min(MAX_TIMEOUT, float(timeout_sec or DEFAULT_TIMEOUT)))

    builtins_module = __import__("builtins")
    safe_builtins = {name: getattr(builtins_module, name) for name in _SAFE_BUILTINS if hasattr(builtins_module, name)}

    payload = json.dumps(
        {"code": code, "builtins": sorted(safe_builtins), "modules": list(_SAFE_MODULES)}
    )

    # The child rebuilds its own builtins dict from the names it is given, so nothing is
    # pickled and nothing is trusted across the boundary.
    bootstrap = _RUNNER.replace(
        'namespace = {"__builtins__": payload["builtins"]}',
        'import builtins as _b\n'
        'namespace = {"__builtins__": {n: getattr(_b, n) for n in payload["builtins"] if hasattr(_b, n)}}',
    )

    with tempfile.TemporaryDirectory() as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", bootstrap],
                input=payload,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                preexec_fn=lambda: _limits(int(timeout) + 1),
                env={"PATH": "", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired as exc:
            return (
                _clamp(exc.stdout or ""),
                "TIMEOUT_EXCEEDED",
                None,
            )

    stdout = _clamp(completed.stdout or "")
    stderr = _clamp(completed.stderr or "")
    value = None

    if completed.returncode != 0 and not stderr.strip():
        # Killed by SIGXCPU or SIGKILL: the CPU limit caught what the wall clock did not.
        if completed.returncode in {-9, -24}:
            return stdout.strip(), "TIMEOUT_EXCEEDED", None
        return stdout.strip(), "EXECUTION_ERROR", None

    if "[result] " in stdout:
        stdout, _, tail = stdout.rpartition("[result] ")
        value = tail.strip()

    return stdout.strip(), stderr.strip(), value


def _clamp(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    body = text or ""

    if len(body) <= limit:
        return body

    return body[:limit] + "\n[Output truncated at 4000 characters]"


# ---------------------------------------------------------------- the decision


def _error(code: str, language: str, detail: str = "") -> SandboxExecutionResult:
    table = _MESSAGES.get(code, _MESSAGES["INVALID_ARGUMENT"])
    key = (language or "en").strip().casefold()[:2]
    message = table.get(key, table["en"])

    return SandboxExecutionResult(
        success=False,
        error_code=code,
        error_message=message,
        stderr=f"{message} {detail}".strip(),
        action_chips=[{"label": "Debug Code", "action": "debug_snippet"}],
    )


def validate_and_execute(call: SandboxToolCall, *, language: str = "en") -> SandboxExecutionResult:
    """Audit, then run, then bound what comes back."""
    started = time.perf_counter()

    if call.tool_name != "execute_python_snippet":
        result = _error("INVALID_ARGUMENT", language, "unknown tool")
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    code = str((call.arguments or {}).get("code") or "")

    if not code.strip():
        result = _error("INVALID_ARGUMENT", language, "empty snippet")
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    safe, problem = audit_ast_safety(code)

    if not safe:
        result = _error("BLOCKED_SECURITY_VIOLATION", language, f"({problem})")
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    try:
        timeout = float((call.arguments or {}).get("timeout_seconds", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT

    stdout, stderr, value = run_isolated(code, timeout)

    if stderr == "TIMEOUT_EXCEEDED":
        result = _error("TIMEOUT_EXCEEDED", language)
        result.stdout = stdout
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    if stderr:
        result = _error("EXECUTION_ERROR", language)
        result.stdout = stdout
        result.stderr = stderr
        result.telemetry_micros = (time.perf_counter() - started) * 1_000_000
        return result

    return SandboxExecutionResult(
        success=True,
        stdout=stdout,
        stderr="",
        return_value=value,
        action_chips=[
            {"label": "Explain Code", "action": "explain_snippet"},
            {"label": "Run New Snippet", "action": "new_snippet"},
        ],
        telemetry_micros=(time.perf_counter() - started) * 1_000_000,
    )
