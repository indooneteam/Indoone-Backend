from app.ai.tool_registry import (
    EMAIL_TOOL_DECLARATIONS,
    GITHUB_TOOL_DECLARATIONS,
    TOOL_SPECS,
    get_tool_spec,
    tool_names,
)
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


def test_tool_names_are_unique() -> None:
    names = [spec.name for spec in TOOL_SPECS]
    assert len(names) == len(set(names))


def test_every_tool_is_fully_described() -> None:
    """A tool cannot be half-added: it needs a description, a parameter schema,
    a valid payload style and an owning module."""
    problems: list[str] = []
    for spec in TOOL_SPECS:
        if not spec.description.strip():
            problems.append(f"{spec.name}: missing description")
        if not isinstance(spec.parameters, dict):
            problems.append(f"{spec.name}: parameters must be a schema object")
        if spec.payload_style not in ("json", "text"):
            problems.append(f"{spec.name}: bad payload_style {spec.payload_style!r}")
        if not spec.module.strip():
            problems.append(f"{spec.name}: missing module")
    assert problems == []


def test_github_declarations_are_derived_and_read_only() -> None:
    """The GitHub declarations must come from the catalogue, not a hand copy."""
    assert GITHUB_TOOL_DECLARATIONS
    for declaration in GITHUB_TOOL_DECLARATIONS:
        spec = get_tool_spec(declaration["name"])
        assert spec is not None
        assert spec.connector_id == "github"
        assert spec.requires_approval is False
        assert declaration["parameters"] == spec.parameters
        assert declaration["description"] == spec.description


def test_email_declarations_are_complete() -> None:
    assert EMAIL_TOOL_DECLARATIONS
    for declaration in EMAIL_TOOL_DECLARATIONS:
        assert declaration["name"]
        assert declaration["description"]
        assert declaration["parameters"]
