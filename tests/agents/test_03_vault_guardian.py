"""Agent 03 - the Vault Guardian.

Two halves, and the second matters as much as the first:

1. What must not leave the machine is removed - identifiers, cards, keys, credentials.
2. What must not be touched is left exactly as it is - a commit hash, a timestamp, a port,
   an order number, a formula, a sentence in Kannada.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

vault = load("03_vault_guardian")


# ---------------------------------------------------------------- 1. clean text passes


@pytest.mark.parametrize(
    "message",
    [
        "when is deepavali this year?",
        "ಇವತ್ತು ದೀಪಾವಳಿ ಯಾವಾಗ? ನನಗೆ ದಿನಾಂಕ ಬೇಕು.",
        "आज मौसम कैसा है? मुझे बताइए।",
        "இன்று தீபாவளி எப்போது? தேதி சொல்லுங்கள்.",
        "ఈరోజు వాతావరణం ఎలా ఉంది? చెప్పండి.",
        "web research maadu jio ipo listing date",
        "list my branches and the last commit",
    ],
)
def test_a_clean_prompt_is_untouched(message: str) -> None:
    verdict = vault.redact(message)

    assert verdict.status == vault.PASS
    assert verdict.redaction_count == 0
    assert verdict.redacted_prompt == message
    assert verdict.detected_entities == []


# ---------------------------------------------------------------- 2. identifiers


@pytest.mark.parametrize(
    ("message", "entity"),
    [
        ("my aadhaar is 2345 6789 0123", "aadhaar"),
        ("aadhaar 234567890123", "aadhaar"),
        ("PAN ABCDE1234F", "pan"),
        ("IFSC SBIN0001234", "ifsc"),
        ("pay me at ravi@okaxis", "upi"),
        ("call 9876543210", "phone"),
        ("call +91 98765 43210", "phone"),
        ("mail ravi.kumar@example.com", "email"),
        ("card 4111 1111 1111 1111", "card"),
    ],
)
def test_an_identifier_is_redacted(message: str, entity: str) -> None:
    verdict = vault.redact(message)

    assert verdict.status == vault.REDACTED
    assert entity in verdict.detected_entities
    assert entity in " ".join(verdict.detected_entities)
    assert message != verdict.redacted_prompt


def test_the_placeholder_names_what_was_removed() -> None:
    verdict = vault.redact("my aadhaar is 2345 6789 0123")

    assert "[Aadhaar Redacted]" in verdict.redacted_prompt
    assert "2345" not in verdict.redacted_prompt


@pytest.mark.parametrize(
    ("message", "entity"),
    [
        ("key sk-abcdefghijklmnop1234567890", "api_key"),
        ("token ghp_abcdefghijklmnopqrstuvwxyz0123", "github_token"),
        ("AKIAIOSFODNN7EXAMPLE", "aws_key"),
        ("postgres://admin:hunter2@db.internal:5432/app", "db_credentials"),
        ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop", "jwt"),
    ],
)
def test_an_infrastructure_secret_is_neutralised(message: str, entity: str) -> None:
    verdict = vault.redact(message)

    assert verdict.status == vault.REDACTED
    assert entity in verdict.detected_entities


def test_database_credentials_go_but_the_shape_stays() -> None:
    verdict = vault.redact("postgres://admin:hunter2@db.internal:5432/app")

    assert "hunter2" not in verdict.redacted_prompt
    assert "db.internal:5432" in verdict.redacted_prompt


# ---------------------------------------------------------------- 3. thinking


@pytest.mark.parametrize(
    "message",
    [
        "commit 632a7f6d09 is on main",
        "the build failed at timestamp 1711900000",
        "listen on port=8000 and 5432",
        "the ratio is 3.14 and 2+2=4",
        "order 1234567890123456 shipped",  # sixteen digits, not a valid card
        "status 404 from the API",
        "in 2026 we ship",
        "hash cb26de7330",
    ],
)
def test_technical_numbers_are_not_redacted(message: str) -> None:
    verdict = vault.redact(message)

    assert verdict.status == vault.PASS, verdict.detected_entities
    assert verdict.redacted_prompt == message


def test_a_card_is_redacted_only_when_the_checksum_agrees() -> None:
    valid = vault.redact("card 4111 1111 1111 1111")
    invalid = vault.redact("card 4111 1111 1111 1112")

    assert valid.status == vault.REDACTED
    assert invalid.status == vault.PASS


def test_code_survives() -> None:
    code = "for i in range(8000):\n    print(i, 98765)\n"

    verdict = vault.redact(code)

    assert verdict.status == vault.PASS
    assert verdict.redacted_prompt == code


# ---------------------------------------------------------------- 4. contract


def test_the_contract_is_complete() -> None:
    verdict = vault.redact("aadhaar 2345 6789 0123 and card 4111 1111 1111 1111")

    assert verdict.status in {vault.PASS, vault.REDACTED}
    assert verdict.original_prompt
    assert verdict.redacted_prompt
    assert isinstance(verdict.detected_entities, list)
    assert verdict.redaction_count == len(
        [entity for entity in verdict.detected_entities]
    ) or verdict.redaction_count >= 1
    assert verdict.telemetry.execution_microseconds >= 0
    assert verdict.telemetry.entities_evaluated > 0


def test_several_entities_are_all_reported() -> None:
    verdict = vault.redact("aadhaar 2345 6789 0123, call 9876543210, mail a@b.com")

    assert set(verdict.detected_entities) >= {"aadhaar", "phone", "email"}
    assert verdict.redaction_count >= 3


# ---------------------------------------------------------------- 5. speed


def test_a_typical_prompt_costs_well_under_a_millisecond() -> None:
    message = "when is deepavali this year and what about the holiday list?"

    started = time.perf_counter()
    for _ in range(500):
        vault.redact(message)
    elapsed_ms = (time.perf_counter() - started) * 1000 / 500

    assert elapsed_ms < 0.5, f"mean latency {elapsed_ms:.3f} ms"


# ---------------------------------------------------------------- 6. wiring


def _auth_headers(user_id: str = "vault-user", secret: str = "x" * 32) -> dict[str, str]:
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


def test_the_model_never_sees_the_raw_identifier(monkeypatch) -> None:
    """The proof that matters: what reaches generate_reply has no Aadhaar number in it."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    monkeypatch.setenv("INDOONE_VAULT_GUARDIAN_ENABLED", "true")

    seen: list[str] = []

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        seen.append(message)
        return "ok"

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"message": "my aadhaar is 2345 6789 0123 and my number is 9876543210"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert seen, "the model should have been called"
    assert "2345" not in seen[0]
    assert "9876543210" not in seen[0]
    assert "[Aadhaar Redacted]" in seen[0]
    assert "[Phone Redacted]" in seen[0]
