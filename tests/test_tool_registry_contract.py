from app.ai.tool_registry import TOOL_SPECS
from app.ai.tools import TOOLS


def test_every_registered_tool_has_runtime_implementation() -> None:
    missing = sorted(spec.name for spec in TOOL_SPECS if spec.name not in TOOLS)
    assert missing == []


def test_connector_tools_have_explicit_capability_metadata() -> None:
    missing = [
        spec.name
        for spec in TOOL_SPECS
        if spec.connector_id and not spec.capability
    ]
    assert missing == []
