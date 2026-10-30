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


def intent() -> ModuleType:
    """Agent 05 - the intent extractor: case markers, action, resource, free-path verdict."""
    return load("05_intent_purity_extractor")


# The circuit breaker is imported by the gateway, which reads it as a module rather than
# calling an accessor: `circuit_breaker.GEMINI`, `circuit_breaker.preferred_order()`. It is
# loaded once here so the gateway does not carry the loading rule.
circuit_breaker = load("06_circuit_breaker")


def treasury() -> ModuleType:
    """Agent 07 - the answer cache: public facts only, bounded, TTL'd."""
    return load("07_treasury_cache")


def condenser() -> ModuleType:
    """Agent 08 - the context condenser: what history this turn actually needs."""
    return load("08_context_condenser")


def arbiter() -> ModuleType:
    """Agent 09 - the token budget arbiter: what a turn may cost, and what is left."""
    return load("09_token_budget_arbiter")


def model_arbiter() -> ModuleType:
    """Agent 10 - the egress arbiter: leak shield, chips, and channel formatting."""
    return load("10_model_arbiter")


def github_supervisor() -> ModuleType:
    """Agent 11 - the GitHub supervisor: the only door to the API."""
    return load("11_github_supervisor")


def mail_supervisor() -> ModuleType:
    """Agent 12 - the mail supervisor: reading is free, sending needs a person."""
    return load("12_zoho_mail_supervisor")


def calendar_supervisor() -> ModuleType:
    """Agent 13 - the calendar supervisor: dates are arithmetic, not memory."""
    return load("13_calendar_supervisor")


def web_supervisor() -> ModuleType:
    """Agent 14 - the web supervisor: the internet, through one narrow door."""
    return load("14_web_search_supervisor")


def code_sandbox() -> ModuleType:
    """Agent 15 - the sandbox: run the arithmetic, not the machine."""
    return load("15_code_sandbox_runner")


def stack_trace_debugger() -> ModuleType:
    """Agent 16 - the debugger: find the bug, not the framework."""
    return load("16_stack_trace_debugger")


def host_sentry() -> ModuleType:
    """Agent 17 - the host sentry: ask the machine, do not command it."""
    return load("17_system_diagnostics_sentry")


def whatsapp_ingress() -> ModuleType:
    """Agent 18 - the WhatsApp door: prove it came from Meta, then flatten it."""
    return load("18_whatsapp_ingress")


def telegram_ingress() -> ModuleType:
    """Agent 19 - the Telegram door: same discipline as WhatsApp, different shape."""
    return load("19_telegram_bot_marshaller")


def instagram_ingress() -> ModuleType:
    """Agent 20 - the Instagram door: a DM has no markdown and a thousand characters."""
    return load("20_instagram_dm_adapter")


def app_streamer() -> ModuleType:
    """Agent 21 - the streamer: tokens as they are produced, nothing after the client leaves."""
    return load("21_app_streamer")


def quota_guard() -> ModuleType:
    """Agent 22 - the quota guard: a free tier is a budget, and a budget needs a ceiling."""
    return load("22_omnichannel_quota_guard")


def pool_watchdog() -> ModuleType:
    """Agent 23 - the pool watchdog: watch the pool, never touch the data."""
    return load("23_postgres_pool_watchdog")


def lru_pruner() -> ModuleType:
    """Agent 24 - the pruner: keep the process alive, and decide what to forget."""
    return load("24_lru_state_pruner")


def cost_ledger() -> ModuleType:
    """Agent 25 - the ledger: what the system spent, and what it did not have to."""
    return load("25_token_cost_ledger")


def circuit_healer() -> ModuleType:
    """Agent 26 - the healer: find out whether Gemini is back, without asking a user."""
    return load("26_circuit_breaker_healer")
