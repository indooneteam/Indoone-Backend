from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable


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


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": _calculate,
}


def run_tool(name: str, payload: str) -> ToolResult:
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(name, f"Unknown tool: {name}", safe=False)
    try:
        return ToolResult(name, tool(payload))
    except Exception as exc:
        return ToolResult(name, f"Tool error: {exc}", safe=False)
