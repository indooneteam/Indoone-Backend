"""Single-purpose agents that guard and operate the system.

Distinct from ``app.ai.agents``: those plan work for a user's request. These watch the
system itself - ingress, health, repair - and answer to an operator, not to a chat turn.

Each file is numbered so the order of the watch is obvious at a glance. A numbered module
name is not importable with ``import``, so ``load()`` reads it by path and caches the
result; callers never need to know that.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_DIRECTORY = Path(__file__).resolve().parent
_cache: dict[str, ModuleType] = {}


def load(name: str) -> ModuleType:
    """Load an agent module by its file stem, e.g. ``load("01_citadel_gatekeeper")``."""
    if name in _cache:
        return _cache[name]

    path = _DIRECTORY / f"{name}.py"

    if not path.is_file():
        raise ImportError(f"no agent module named {name!r} in {_DIRECTORY}")

    spec = importlib.util.spec_from_file_location(f"app.agents.{name}", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"agent module {name!r} could not be loaded")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    _cache[name] = module
    return module


def gatekeeper() -> ModuleType:
    """Agent 01 - the ingress defender."""
    return load("01_citadel_gatekeeper")


def sentinel() -> ModuleType:
    """Agent 02 - the intent reader: jailbreak, system leak, exploit, obfuscation."""
    return load("02_inquisitor_sentinel")


def vault() -> ModuleType:
    """Agent 03 - the privacy guardian: identifiers, cards, keys, credentials."""
    return load("03_vault_guardian")


def marshal() -> ModuleType:
    """Agent 04 - the language marshal: script, dialect, canonical verb, register."""
    return load("04_indic_polyglot_marshal")
