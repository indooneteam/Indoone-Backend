import pytest

from app.ai.agent_contracts import ToolExecutionContract, ToolInvocation, serialize_tool_result
from app.ai.tools import ToolResult


def test_tool_invocation_normalizes_registered_tool() -> None:
    invocation = ToolInvocation(tool="  CALCULATOR ", payload="2+3", index=1)
    assert invocation.tool == "calculator"


def test_tool_invocation_rejects_unknown_or_invalid() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        ToolInvocation(tool="missing", payload="x", index=1)
    with pytest.raises(ValueError, match="index"):
        ToolInvocation(tool="calculator", payload="1", index=0)


def test_tool_execution_contract_serializes_metadata() -> None:
    result = ToolResult("calculator", "5", safe=True, truncated=False)
    contract = ToolExecutionContract(result=result, retry_count=1, elapsed_ms=12)
    assert contract.to_dict()["retry_count"] == 1
    assert contract.to_dict()["elapsed_ms"] == 12
    assert serialize_tool_result(result)["truncated"] is False
