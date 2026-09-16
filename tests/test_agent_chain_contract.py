from app.ai.agent import _chain_payload, _unresolved_chain_references, execute_agent
from app.ai.tools import ToolResult


def _result(output: str) -> ToolResult:
    return ToolResult(name="text_stats", output=output)


def test_chain_payload_does_not_reprocess_placeholder_tokens_from_output() -> None:
    results = (_result("literal $result1"),)
    assert _chain_payload("value=$last", results) == "value=literal $result1"


def test_chain_payload_replaces_all_references_in_one_pass() -> None:
    results = (_result("first"), _result("second"))
    payload = "a=$result1 b=$last c=$result2"
    assert _chain_payload(payload, results) == "a=first b=second c=second"


def test_chain_payload_keeps_unknown_references_unchanged() -> None:
    results = (_result("first"),)
    assert _chain_payload("$result9/$last", results) == "$result9/first"


def test_unresolved_chain_reference_is_detected_before_tool_execution() -> None:
    results = (_result("first"),)
    assert _unresolved_chain_references("$result2 * 4", results) == ("$result2",)
    assert _unresolved_chain_references("$last * 4", results) == ()


def test_agent_blocks_execution_for_unresolved_chain_reference(monkeypatch) -> None:
    calls: list[str] = []

    def should_not_run(name: str, payload: str):
        calls.append(payload)
        return ToolResult(name, "unexpected", safe=True, retryable=False)

    monkeypatch.setattr("app.ai.agent.run_tool", should_not_run)
    execution = execute_agent("calculate $result2 + 4")
    assert execution.results
    assert execution.results[0].safe is False
    assert execution.results[0].retryable is False
    assert "$result2" in execution.results[0].output
    assert calls == []
