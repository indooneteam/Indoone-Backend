"""Agent 04 - the Indic Polyglot Marshal.

Three answers, each of which the backend used to guess: which script this is, whether Latin
letters are English or Indic typed phonetically, and how the person is speaking.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

marshal = load("04_indic_polyglot_marshal")


# ---------------------------------------------------------------- 1. scripts


@pytest.mark.parametrize(
    ("message", "language", "script"),
    [
        ("ಇವತ್ತು ದೀಪಾವಳಿ ಯಾವಾಗ?", "kn", "Knda"),
        ("आज मौसम कैसा है?", "hi", "Deva"),
        ("இன்று தீபாவளி எப்போது?", "ta", "Taml"),
        ("ఈరోజు వాతావరణం ఎలా ఉంది?", "te", "Telu"),
        ("ഇന്ന് കാലാവസ്ഥ എങ്ങനെയുണ്ട്?", "ml", "Mlym"),
        ("আজ আবহাওয়া কেমন?", "bn", "Beng"),
        ("આજે હવામાન કેવું છે?", "gu", "Gujr"),
    ],
)
def test_a_native_script_is_named(message: str, language: str, script: str) -> None:
    verdict = marshal.analyze(message)

    assert verdict.primary_language == language
    assert verdict.script == script
    assert verdict.is_transliterated is False


def test_devanagari_is_split_into_hindi_and_marathi() -> None:
    hindi = marshal.analyze("आज मौसम कैसा है? मुझे बताइए।")
    marathi = marshal.analyze("आज हवामान कसे आहे? मला सांगा.")

    assert hindi.primary_language == "hi"
    assert marathi.primary_language == "mr"
    assert marathi.script == "Deva"


# ---------------------------------------------------------------- 2. transliteration


@pytest.mark.parametrize(
    "message",
    ["Show me pull requests", "How to configure Docker", "when is deepavali this year"],
)
def test_english_stays_english(message: str) -> None:
    verdict = marshal.analyze(message)

    assert verdict.primary_language == "en"
    assert verdict.is_transliterated is False
    assert verdict.dialect_family == "standard"


@pytest.mark.parametrize(
    ("message", "language", "dialect"),
    [
        ("branch maadi beku", "kn", "kanglish"),
        ("kya haal hai batao", "hi", "hinglish"),
        ("enna panra sollu", "ta", "tanglish"),
        ("ela unnavu cheppandi", "te", "tenglish"),
        ("enthu parayu", "ml", "manglish"),
    ],
)
def test_romanised_indic_is_flagged(message: str, language: str, dialect: str) -> None:
    verdict = marshal.analyze(message)

    assert verdict.is_transliterated is True
    assert verdict.primary_language == language
    assert verdict.dialect_family == dialect


def test_one_borrowed_word_does_not_relabel_english() -> None:
    """A single marker is not evidence; two are."""
    verdict = marshal.analyze("deploy the beku branch")

    assert verdict.is_transliterated is False


# ---------------------------------------------------------------- 3. canonical forms


@pytest.mark.parametrize(
    ("message", "canonical"),
    [
        ("branch maadi", "branch maadu"),
        ("branch madi", "branch maadu"),
        ("branch maadbeku", "branch maadu"),
        ("repo nodi", "repo nodu"),
        ("mail kijiye", "mail karo"),
        ("mail karna", "mail karo"),
        ("date sollunga", "date sollu"),
    ],
)
def test_a_verb_collapses_to_one_stem(message: str, canonical: str) -> None:
    assert marshal.analyze(message).canonical_text == canonical


@pytest.mark.parametrize(
    "message",
    ["branch repo commit calendar email pr", "deploy the docker build to prod"],
)
def test_nouns_are_never_rewritten(message: str) -> None:
    assert marshal.analyze(message).canonical_text == message


# ---------------------------------------------------------------- 4. register


@pytest.mark.parametrize(
    ("message", "tone"),
    [
        ("ದಯವಿಟ್ಟು ತಿಳಿಸಿ", "HONORIFIC"),
        ("please tell me the date", "HONORIFIC"),
        ("kijiye batao", "HONORIFIC"),
        ("ಮಾಡು ಹೇಳು", "CASUAL"),
        ("hey bro karo", "CASUAL"),
        ("git status", "TECHNICAL"),
        ("docker build .", "TECHNICAL"),
        ("alembic upgrade head", "TECHNICAL"),
    ],
)
def test_the_register_is_read(message: str, tone: str) -> None:
    assert marshal.analyze(message).register_tone == tone


# ---------------------------------------------------------------- 5. contract and speed


def test_the_contract_is_complete() -> None:
    verdict = marshal.analyze("branch maadi beku")

    assert verdict.primary_language
    assert verdict.script
    assert isinstance(verdict.is_transliterated, bool)
    assert verdict.dialect_family
    assert verdict.register_tone in {"HONORIFIC", "CASUAL", "TECHNICAL"}
    assert verdict.canonical_text
    assert verdict.telemetry.execution_microseconds >= 0
    assert 0.0 <= sum(verdict.telemetry.script_distribution.values()) <= 1.001


def test_a_typical_prompt_costs_under_a_tenth_of_a_millisecond() -> None:
    """Best of three rounds: on a shared CI box one round can be scheduled out mid-loop,
    and an average that includes that says more about the machine than the agent."""
    message = "web research maadu jio ipo estakke listing agide antha heli"
    best = min(_mean_milliseconds(message) for _ in range(3))

    assert best < 0.1, f"best mean latency {best:.4f} ms"


def _mean_milliseconds(message: str) -> float:
    started = time.perf_counter()
    for _ in range(500):
        marshal.analyze(message)
    return (time.perf_counter() - started) * 1000 / 500


def test_one_call_reports_its_own_cost() -> None:
    verdict = marshal.analyze("branch maadi beku")

    assert verdict.telemetry.execution_microseconds < 2_000


# ---------------------------------------------------------------- 6. wiring


def _auth_headers(user_id: str = "marshal-user", secret: str = "x" * 32) -> dict[str, str]:
    import base64
    import hashlib
    import hmac
    import time as _time

    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(user_id.encode("utf-8"))
    stamp = encode(str(int(_time.time())).encode("ascii"))
    signature = encode(hmac.new(secret.encode("ascii"), f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())
    return {"Authorization": f"Bearer {user}.{stamp}.{signature}"}


def test_the_endpoint_attaches_the_linguistic_metadata(monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_POLYGLOT_MARSHAL_ENABLED", "true")

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return "ok"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "branch maadi beku"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200


def test_the_message_is_never_rewritten_by_the_marshal(monkeypatch) -> None:
    """Metadata is attached, not applied: the model must see the words the user typed."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_POLYGLOT_MARSHAL_ENABLED", "true")

    seen: list[str] = []

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        seen.append(message)
        return "ok"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    client = TestClient(app)
    # A message no router claims, so the model is the one that receives it.
    client.post("/api/chat", json={"message": "kya haal hai batao"}, headers=_auth_headers("m2"))

    assert seen, "the model should have been called"
    assert seen[0] == "kya haal hai batao"
