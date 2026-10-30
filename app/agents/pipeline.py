"""The frontline: five agents, one call, one order.

Each agent was built to be read on its own. This is where they are wired, so the order lives
in one place instead of five lines in an endpoint that nobody dares reorder:

    01 gatekeeper  -> how much, how often, what shape      (429 with Retry-After)
    02 sentinel    -> what is being asked, and is it an attack (400, in the user's language)
    03 vault       -> what must not leave the machine      (redacted prompt)
    04 marshal     -> which language, which dialect, which tone
    05 intent      -> which action, which resource, is a model needed at all

Two rules make this a cutover rather than a rewrite:

* **A refusal is typed, not raised.** The pipeline returns what the caller should send -
  status, message, retry-after - and the endpoint raises. A library that raises HTTP
  exceptions cannot be used from a worker, a channel handler or a test without a server.
* **It annotates; it does not route.** The routers in ``chat.py`` already answer the
  questions agent 05 recognises, and a second dispatcher that also answers them is how one
  message ends up claimed twice. The verdict travels with the request so the pipeline can
  say what it expected, and the endpoint can log the gap when no router agreed.

Order matters and is not arbitrary: size before content (a flood is refused before it is
scanned), intent before privacy (an attack is refused rather than redacted), privacy before
language (the marshal reads what will actually be sent), language before intent (the intent
extractor works on cleaned text).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import ModuleType

from app.agents import gatekeeper as _gatekeeper
from app.agents import intent as _intent
from app.agents import marshal as _marshal
from app.agents import sentinel as _sentinel
from app.agents import vault as _vault

# The budget for all five together. Each agent has its own tighter bound; this is the one
# that matters to a request.
PIPELINE_BUDGET_MILLISECONDS = 0.25


@dataclass(slots=True, frozen=True)
class Refusal:
    """What to send back, and why - in a shape a caller can act on without a server."""

    status_code: int
    detail: str
    retry_after: int = 0
    agent: str = ""
    # The machine-readable reason, for a client that wants to act rather than display. The
    # detail is for the person; this is for the code that receives it.
    code: str = ""


@dataclass(slots=True, frozen=True)
class FrontlineVerdict:
    """Everything the request earned on its way in."""

    prompt: str
    refusal: Refusal | None = None
    linguistics: object | None = None
    intent: object | None = None
    redactions: int = 0
    elapsed_microseconds: int = 0
    ran: tuple[str, ...] = field(default_factory=tuple)

    @property
    def refused(self) -> bool:
        return self.refusal is not None

    @property
    def fast_path_target(self) -> str | None:
        """The free path agent 05 expects, when it is sure enough to expect one."""
        verdict = self.intent
        return getattr(verdict, "fast_path_target", None) if verdict is not None else None


def _module(loader, name: str) -> ModuleType | None:
    try:
        return loader()
    except Exception:  # an agent that cannot load must not take the request with it
        return None


def process(
    message: str,
    *,
    subject: str = "",
    client_ip: str = "",
    now: float | None = None,
) -> FrontlineVerdict:
    """Run the five agents in order and report what they concluded.

    Never raises for a refusal: the caller decides what a refusal means for it.
    """
    started = time.perf_counter()
    prompt = message or ""
    ran: list[str] = []

    # 01 - the ingress defender. Size and tempo first: a flood is refused before anything
    # downstream spends a microsecond on it.
    gate = _module(_gatekeeper, "01_citadel_gatekeeper")

    if gate is not None and gate.enabled():
        decision = gate.evaluate(subject=subject, payload=prompt, client_ip=client_ip, now=now)
        ran.append("01_gatekeeper")

        if not decision.allowed:
            return FrontlineVerdict(
                prompt=prompt,
                refusal=Refusal(
                    status_code=429,
                    code="RATE_LIMITED",
                    detail=decision.client_message or "Too many requests. Try again shortly.",
                    retry_after=max(1, decision.retry_after),
                    agent="01_gatekeeper",
                ),
                elapsed_microseconds=int((time.perf_counter() - started) * 1_000_000),
                ran=tuple(ran),
            )

    # 02 - the intent reader. An attack is refused, not sanitised: the user is told in their
    # own language and the provider is never called.
    guard = _module(_sentinel, "02_inquisitor_sentinel")

    if guard is not None and guard.enabled():
        verdict = guard.evaluate(prompt)
        ran.append("02_sentinel")

        if verdict.rejected:
            return FrontlineVerdict(
                prompt=prompt,
                refusal=Refusal(
                    status_code=400,
                    detail=verdict.client_message.get("primary") or verdict.client_message.get("en", ""),
                    agent="02_sentinel",
                    code="SECURITY_REJECTED",
                ),
                elapsed_microseconds=int((time.perf_counter() - started) * 1_000_000),
                ran=tuple(ran),
            )

        prompt = verdict.safe_prompt or prompt

    # 03 - the privacy guardian. Everything after this point reads the redacted text, so a
    # leak downstream cannot happen by forgetting to redact.
    keep = _module(_vault, "03_vault_guardian")
    redactions = 0

    if keep is not None and keep.enabled():
        vault_verdict = keep.redact(prompt)
        ran.append("03_vault")
        redactions = vault_verdict.redaction_count
        prompt = vault_verdict.redacted_prompt

    # 04 - the language marshal. Reads what will actually be sent.
    language = _module(_marshal, "04_indic_polyglot_marshal")
    linguistics = None

    if language is not None and language.enabled():
        linguistics = language.analyze(prompt)
        ran.append("04_marshal")

    # 05 - the intent extractor. Reports whether a model is needed; it does not dispatch.
    extractor = _module(_intent, "05_intent_purity_extractor")
    intent_verdict = None

    if extractor is not None and extractor.enabled():
        intent_verdict = extractor.extract(prompt)
        ran.append("05_intent")

    return FrontlineVerdict(
        prompt=prompt,
        linguistics=linguistics,
        intent=intent_verdict,
        redactions=redactions,
        elapsed_microseconds=int((time.perf_counter() - started) * 1_000_000),
        ran=tuple(ran),
    )
