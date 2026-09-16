from app.ai.agent import _chain_payload
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
