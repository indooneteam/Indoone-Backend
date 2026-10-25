"""The coding tools must be callable end to end from their own declaration.

The registry is the single source of truth for what the model may send. These
tests run each coding tool the way the dispatcher does -- build the payload from
the declared schema, then execute -- so a payload_style or schema drift between
the registry and the executor fails here instead of silently at runtime.
"""
from __future__ import annotations

import json

import pytest

from app.ai.agents.agent import build_agent_steps
from app.ai.agents.coding import SUPPORTED_LANGUAGES, detect_language
from app.ai.agents.orchestrator import plan_request
from app.ai.execution.tool_loop import _core_payload
from app.ai.execution.tool_registry import get_tool_spec
from app.ai.execution.tools import run_tool

CODING_TOOLS = ("code_analysis", "code_fix_suggestions", "code_transform", "sandbox_execution")


@pytest.mark.parametrize("name", CODING_TOOLS)
def test_coding_tools_declare_json_language_and_code(name: str) -> None:
    spec = get_tool_spec(name)
    assert spec is not None
    assert spec.payload_style == "json"
    properties = spec.parameters["properties"]
    assert "language" in properties
    assert "code" in properties
    assert {"language", "code"} <= set(spec.parameters["required"])


@pytest.mark.parametrize("name", CODING_TOOLS)
def test_coding_tool_runs_from_its_declared_arguments(name: str) -> None:
    spec = get_tool_spec(name)
    assert spec is not None
    payload = _core_payload(spec, {"language": "python", "code": "x = 1  "})
    result = run_tool(name, payload)
    assert result.safe is True, result.output
    assert json.loads(result.output)


@pytest.mark.parametrize("name", CODING_TOOLS)
def test_coding_tool_declared_language_enum_matches_the_executor(name: str) -> None:
    spec = get_tool_spec(name)
    assert spec is not None
    declared = set(spec.parameters["properties"]["language"]["enum"])
    assert declared == set(SUPPORTED_LANGUAGES)


def test_detect_language_maps_common_markers() -> None:
    assert detect_language("def add(a, b):\n    return a + b") == "python"
    assert detect_language("function add(a, b) { return a + b; }") == "javascript"
    assert detect_language("package main\nfunc main() {}") == "go"
    assert detect_language('fn main() { println!("hi"); }') == "rust"
    assert detect_language("x = 1") == "python"


def test_agent_wraps_coding_commands_in_a_json_payload() -> None:
    steps = build_agent_steps("analyze code: def add(a, b): return a + b")
    assert len(steps) == 1
    assert steps[0].tool == "code_analysis"
    result = run_tool(steps[0].tool, steps[0].payload)
    assert result.safe is True, result.output


def test_planner_wraps_coding_commands_in_a_json_payload() -> None:
    plan = plan_request("review code: print('hi')")
    assert plan.tool == "code_analysis"
    assert plan.tool_payload is not None
    result = run_tool(plan.tool, plan.tool_payload)
    assert result.safe is True, result.output
