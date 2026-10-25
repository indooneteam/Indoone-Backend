"""The unit in the repository must name an entry point that exists.

This test exists because it did not, and the service went down for it. When scripts/
was grouped into folders the entry point moved, every reference was updated except the
one in deploy/indoone-backend.service - which is not a Python file, so the sweep did not
look at it. The box then restarted every five seconds with a missing module error.

A unit file that names a module nobody has is a silent, total outage. This makes it a
red build instead.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIT_FILE = PROJECT_ROOT / "deploy" / "indoone-backend.service"
DEPLOY_DOC = PROJECT_ROOT / "deploy" / "DEPLOY.md"
DEPLOY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "deploy.sh"

_MODULE_ARGUMENT = re.compile(r"-m\s+([A-Za-z_][A-Za-z0-9_.]*)")


def _unit_text() -> str:
    assert UNIT_FILE.is_file(), "deploy/indoone-backend.service is missing"
    return UNIT_FILE.read_text(encoding="utf-8")


def test_the_unit_names_the_expected_entry_point() -> None:
    executables = [line for line in _unit_text().splitlines() if line.startswith("ExecStart=")]
    assert len(executables) == 1, f"expected one ExecStart, found {executables}"
    match = _MODULE_ARGUMENT.search(executables[0])
    assert match is not None, f"ExecStart does not name a module: {executables[0]}"
    assert match.group(1) == "scripts.ops.run_server"


def test_that_module_can_actually_be_imported() -> None:
    match = _MODULE_ARGUMENT.search(_unit_text())
    assert match is not None
    module = match.group(1)
    assert importlib.util.find_spec(module) is not None, (
        f"{UNIT_FILE.name} names {module}, but that module does not exist. "
        "The service would restart forever."
    )


def test_the_deploy_doc_and_script_agree_with_the_unit() -> None:
    match = _MODULE_ARGUMENT.search(_unit_text())
    assert match is not None
    module = match.group(1)
    documented = DEPLOY_DOC.read_text(encoding="utf-8")

    assert "run_server" in documented and "scripts/ops" in documented, (
        f"deploy/DEPLOY.md does not point at the entry point the unit names ({module})"
    )
    assert DEPLOY_SCRIPT.is_file(), "scripts/ops/deploy.sh is missing"
    assert "deploy/indoone-backend.service" in DEPLOY_SCRIPT.read_text(encoding="utf-8"), (
        "the deploy script must install the unit from the repository, or a unit change "
        "never reaches the box"
    )
