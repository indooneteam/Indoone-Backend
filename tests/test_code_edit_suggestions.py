from __future__ import annotations

import pytest

from app.plugins.github.edit_suggestions import (
    build_code_edit_prompt,
    parse_code_edit_suggestion,
)


def test_parse_code_edit_suggestion_returns_complete_file() -> None:
    raw = "```python\ndef greet(name):\n    return f\"Hi {name}\"\n```"
    assert parse_code_edit_suggestion(raw) == 'def greet(name):\n    return f"Hi {name}"'


def test_parse_code_edit_suggestion_preserves_markdown_fences_inside_file() -> None:
    raw = "```markdown\n# Guide\n\n```python\nprint('hello')\n```\n```"
    assert parse_code_edit_suggestion(raw) == "# Guide\n\n```python\nprint('hello')\n```"


@pytest.mark.parametrize(
    "raw",
    [
        "Here is your code:\n```python\nprint('hi')\n```",
        "```python\nprint('hi')\n```\nExplanation",
        "print('hi')",
        "```python\n\n```",
    ],
)
def test_parse_code_edit_suggestion_rejects_incomplete_or_extra_text(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_code_edit_suggestion(raw)


def test_parse_code_edit_suggestion_rejects_oversized_file() -> None:
    raw = "```text\n" + ("x" * 50_001) + "\n```"
    with pytest.raises(ValueError, match="limit"):
        parse_code_edit_suggestion(raw)


def test_parse_code_edit_suggestion_rejects_embedded_github_token() -> None:
    raw = "```python\nTOKEN = 'ghp_" + ("A" * 36) + "'\n```"
    with pytest.raises(ValueError, match="secret"):
        parse_code_edit_suggestion(raw)


def test_prompt_marks_source_as_untrusted_and_requires_full_file() -> None:
    prompt = build_code_edit_prompt(
        "owner/repo", "src/app.py", "Add a greeting", "print('old')"
    )
    assert "untrusted data" in prompt
    assert "COMPLETE resulting file" in prompt
    assert "src/app.py" in prompt
    assert "print('old')" in prompt
