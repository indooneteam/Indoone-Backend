"""Every environment variable the app reads must be documented in .env.example.

This keeps deployment setup correct: a missing variable is a silent misconfig,
so a new one cannot be added to the code without also being documented.
"""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = PROJECT_ROOT / "app"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

_PATTERNS = (
    re.compile(r'os\.getenv\(\s*"([A-Z0-9_]+)"'),
    re.compile(r'os\.environ\[\s*"([A-Z0-9_]+)"'),
    re.compile(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"'),
)


def _referenced_names() -> set[str]:
    names: set[str] = set()
    for path in APP_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _PATTERNS:
            names.update(pattern.findall(text))
    return names


def _documented_names() -> set[str]:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, re.MULTILINE))


def test_every_env_var_used_by_the_app_is_documented() -> None:
    missing = sorted(_referenced_names() - _documented_names())
    assert missing == [], f"undocumented environment variables: {missing}"
