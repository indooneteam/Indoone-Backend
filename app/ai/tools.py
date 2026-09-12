from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Callable

from app.ai.coding import code_analysis_tool


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: str
    safe: bool = True


def _calculate(expression: str) -> str:
    tree = ast.parse(expression, mode="eval")
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def walk(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("Only numeric arithmetic is allowed")

    return str(walk(tree.body))


def _text_stats(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    words = normalized.split()
    lines = normalized.splitlines() if normalized else []
    return json.dumps(
        {
            "characters": len(text),
            "words": len(words),
            "lines": len(lines),
        },
        sort_keys=True,
    )


def _json_summary(payload: str) -> str:
    data = json.loads(payload)
    if isinstance(data, dict):
        return json.dumps(
            {"type": "object", "keys": len(data), "key_names": sorted(str(key) for key in data)[:50]},
            ensure_ascii=False,
            sort_keys=True,
        )
    if isinstance(data, list):
        return json.dumps({"type": "array", "items": len(data)}, sort_keys=True)
    if data is None:
        return json.dumps({"type": "null"}, sort_keys=True)
    return json.dumps({"type": type(data).__name__}, sort_keys=True)


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": _calculate,
    "text_stats": _text_stats,
    "json_summary": _json_summary,
    "code_analysis": code_analysis_tool,
}


def run_tool(name: str, payload: str) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(name, f"Unknown tool: {name}", safe=False)
    try:
        return ToolResult(name, tool(payload))
    except Exception as exc:
        return ToolResult(name, f"Tool error: {exc}", safe=False)
