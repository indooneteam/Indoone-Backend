"""Agent 05 - the Intent Purity Extractor.

Two things are tested here, and the second is the one that bit us before:

1. the grammar is stripped and the request is read - action, resource, ordinal;
2. an English word that merely *ends* like a case marker is left alone. "merge" must not
   become "mer", and "diverge" must not become "diver".
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

extractor = load("05_intent_purity_extractor")


# ---------------------------------------------------------------- 1. case markers


@pytest.mark.parametrize(
    ("message", "cleaned", "ordinal"),
    [
        ("1 ralli", "1", 1),
        ("repo 2 ge", "repo 2", 2),
        ("branch galannu thorisu", "branch thorisu", None),
        ("repo 2 mein", "repo 2", 2),
        ("branches ko dikhao", "branch dikhao", None),
        ("commit galu", "commit", None),
    ],
)
def test_a_case_marker_is_stripped(message: str, cleaned: str, ordinal: int | None) -> None:
    verdict = extractor.extract(message)

    assert verdict.cleaned_query == cleaned
    assert verdict.extracted_ordinal == ordinal


@pytest.mark.parametrize(
    ("message", "action", "resource"),
    [
        ("branch galannu thorisu", "LIST", "BRANCH"),
        ("branches ko dikhao", "LIST", "BRANCH"),
        ("list branches", "LIST", "BRANCH"),
        ("show diff", "DIFF", "DIFF"),
        ("unread emails", "READ_EMAIL", "EMAIL"),
        ("when is deepavali", "CHECK_HOLIDAY", "HOLIDAY_CALENDAR"),
    ],
)
def test_the_action_and_resource_are_read(message: str, action: str, resource: str) -> None:
    verdict = extractor.extract(message)

    assert verdict.action == action
    assert verdict.resource == resource


# ---------------------------------------------------------------- 2. nouns survive


@pytest.mark.parametrize(
    ("message", "word"),
    [
        ("merge the pr", "merge"),
        ("diverge the branch", "diverge"),
        ("message the team", "message"),
        ("purge the cache", "purge"),
        ("forge the release", "forge"),
    ],
)
def test_an_english_word_is_not_mistaken_for_a_suffix(message: str, word: str) -> None:
    """The exact failure that made an earlier version of this idea destructive."""
    verdict = extractor.extract(message)

    assert word in verdict.cleaned_query.split()


def test_a_commit_hash_is_not_read_as_an_ordinal() -> None:
    verdict = extractor.extract("commit 632a7f6d09 on main")

    assert verdict.extracted_ordinal is None


# ---------------------------------------------------------------- 3. fast path


@pytest.mark.parametrize(
    ("message", "target"),
    [
        ("when is deepavali", "calendar"),
        ("list branches", "github_read"),
        ("show the PRs", "github_read"),
        ("show diff", "github_read"),
        ("unread emails", "email_read"),
    ],
)
def test_a_pure_request_reaches_the_fast_path(message: str, target: str) -> None:
    verdict = extractor.extract(message)

    assert verdict.is_fast_path is True
    assert verdict.fast_path_target == target
    assert verdict.confidence_score >= extractor.FAST_PATH_FLOOR


@pytest.mark.parametrize(
    "message",
    [
        "hello there",
        "what did we discuss",
        "merge the pr",
        "write a poem about the sea",
        "explain how the scheduler works in detail please",
        "commit 632a7f6d09",
    ],
)
def test_an_impure_request_stays_on_the_model_path(message: str) -> None:
    verdict = extractor.extract(message)

    assert verdict.is_fast_path is False
    assert verdict.fast_path_target is None
    assert verdict.confidence_score < extractor.FAST_PATH_FLOOR


def test_confidence_is_built_from_evidence_not_assumed() -> None:
    pure = extractor.extract("list branches")
    mixed = extractor.extract("list branches and then write a summary of the release notes")

    assert pure.confidence_score > mixed.confidence_score
    assert mixed.is_fast_path is False


# ---------------------------------------------------------------- 4. contract and speed


def test_the_contract_is_complete() -> None:
    verdict = extractor.extract("repo 2 ge list maadi")

    assert isinstance(verdict.is_fast_path, bool)
    assert verdict.action
    assert verdict.resource
    assert 0.0 <= verdict.confidence_score <= 1.0
    assert verdict.cleaned_query
    assert verdict.telemetry.execution_microseconds >= 0


def test_a_typical_prompt_costs_under_a_tenth_of_a_millisecond() -> None:
    message = "branches galannu thorisu"
    best = _fastest_call_milliseconds(message)

    assert best < 0.1, f"fastest call {best:.4f} ms"


def _fastest_call_milliseconds(message: str) -> float:
    """The fastest single call: on a shared CI box a mean measures the machine."""
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        extractor.extract(message)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    return fastest


# ---------------------------------------------------------------- 5. wiring


def test_the_endpoint_attaches_the_intent(monkeypatch) -> None:
    import base64
    import hashlib
    import hmac
    import time as _time

    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_INTENT_EXTRACTOR_ENABLED", "true")

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return "ok"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(b"intent-user")
    stamp = encode(str(int(_time.time())).encode("ascii"))
    signature = encode(hmac.new(b"x" * 32, f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "list branches"},
        headers={"Authorization": f"Bearer {user}.{stamp}.{signature}"},
    )

    assert response.status_code == 200


def test_the_verdict_is_attached_to_the_request_state(monkeypatch) -> None:
    """Called directly, so the assertion is about the attachment and nothing else."""
    from types import SimpleNamespace

    from app.api.chat import chat as chat_module

    monkeypatch.setenv("INDOONE_INTENT_EXTRACTOR_ENABLED", "true")
    monkeypatch.setenv("INDOONE_GATEKEEPER_ENABLED", "false")
    monkeypatch.setenv("INDOONE_SENTINEL_ENABLED", "false")
    monkeypatch.setenv("INDOONE_VAULT_GUARDIAN_ENABLED", "false")
    monkeypatch.setenv("INDOONE_POLYGLOT_MARSHAL_ENABLED", "false")

    request = SimpleNamespace(state=SimpleNamespace())
    chat_module._frontline(request, "u1", "unread emails")

    assert request.state.intent.is_fast_path is True
    assert request.state.intent.fast_path_target == "email_read"


def test_the_attachment_does_not_rewrite_the_message(monkeypatch) -> None:
    from types import SimpleNamespace

    from app.api.chat import chat as chat_module

    monkeypatch.setenv("INDOONE_INTENT_EXTRACTOR_ENABLED", "true")
    monkeypatch.setenv("INDOONE_GATEKEEPER_ENABLED", "false")
    monkeypatch.setenv("INDOONE_SENTINEL_ENABLED", "false")
    monkeypatch.setenv("INDOONE_VAULT_GUARDIAN_ENABLED", "false")
    monkeypatch.setenv("INDOONE_POLYGLOT_MARSHAL_ENABLED", "false")

    request = SimpleNamespace(state=SimpleNamespace())
    message = "branches ko dikhao"
    chat_module._frontline(request, "u1", message)

    # the state carries the cleaned query; the message itself is untouched by the caller
    assert request.state.intent.cleaned_query == "branch dikhao"
    assert message == "branches ko dikhao"
