"""Agent 10 - the egress arbiter.

This is the only agent whose mistakes the user sees directly, so the tests are about what
the person receives: no scaffolding, chips they can press, and a message that renders on the
channel it was sent to.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

arbiter = load("10_model_arbiter")


# ---------------------------------------------------------------- 1. the shield


def test_a_clean_answer_passes_untouched() -> None:
    verdict = arbiter.synthesize("Kannada Rajyotsava is on 1 November.", "app")

    assert verdict.leaks_redacted is False
    assert verdict.formatted_text == "Kannada Rajyotsava is on 1 November."


def test_a_leaked_preamble_is_removed_and_the_answer_kept() -> None:
    leaked = (
        "You are a helpful AI assistant.\n"
        "System instructions: answer politely.\n"
        "\n"
        "Kannada Rajyotsava is on 1 November."
    )

    verdict = arbiter.synthesize(leaked, "app")

    assert verdict.leaks_redacted is True
    assert verdict.formatted_text == "Kannada Rajyotsava is on 1 November."
    assert "helpful AI assistant" not in verdict.formatted_text


def test_a_stray_prompt_tag_does_not_survive() -> None:
    verdict = arbiter.synthesize("Here it is <prompt_context>inner</prompt_context> done.", "app")

    assert verdict.leaks_redacted is True
    assert "<prompt_context>" not in verdict.formatted_text


def test_a_reply_that_is_only_scaffolding_becomes_empty() -> None:
    verdict = arbiter.synthesize("You are a helpful AI assistant.\nSystem instructions: be nice.", "app")

    assert verdict.leaks_redacted is True
    assert verdict.formatted_text == ""


def test_a_technical_mention_mid_answer_is_content_not_a_leak() -> None:
    """\"system instructions\" inside an explanation is the answer, not scaffolding."""
    answer = "The agent reads the system instructions first, then the user message."
    verdict = arbiter.synthesize(answer, "app")

    assert verdict.leaks_redacted is False
    assert verdict.formatted_text == answer


# ---------------------------------------------------------------- 2. the chips


def test_chips_are_inferred_from_what_the_answer_is_about() -> None:
    verdict = arbiter.synthesize("You have three branches in indoone.", "app")

    labels = [chip["label"] for chip in verdict.action_chips]
    assert labels == ["View Diff", "Check PRs"]


def test_a_holiday_answer_offers_the_next_holiday() -> None:
    verdict = arbiter.synthesize("Deepavali is on 8 November.", "app")

    assert verdict.action_chips == [{"label": "Next Holiday", "action": "next_holiday"}]


def test_a_tool_that_emitted_chips_wins_over_a_guess() -> None:
    """A guess must never be offered over a fact the tool already knew."""
    emitted = [{"label": "Open PR 12", "action": "open_pr:12"}]

    verdict = arbiter.synthesize("You have three branches.", "app", existing_chips=emitted)

    assert verdict.action_chips == emitted


def test_an_ordinary_answer_gets_no_chips() -> None:
    verdict = arbiter.synthesize("Docker is a container runtime.", "app")

    assert verdict.action_chips == []


# ---------------------------------------------------------------- 3. the channels


def test_whatsapp_turns_a_table_into_lines_and_stars_bold() -> None:
    table = "**Branches**\n\n| Branch | Status |\n|---|---|\n| feat/auth | open |\n| fix/x | merged |"

    verdict = arbiter.synthesize(table, "whatsapp")
    text = verdict.formatted_text

    assert "|" not in text, "a table does not render on WhatsApp"
    assert "*Branches*" in text
    assert "- feat/auth — Status: open" in text


def test_whatsapp_respects_its_length_limit() -> None:
    long_text = "word " * 3000

    payload = arbiter.synthesize(long_text, "whatsapp").channel_payload

    chunks = payload["chunks"]
    assert len(chunks) > 1
    assert all(len(chunk) <= arbiter.WHATSAPP_LIMIT for chunk in chunks)


def test_whatsapp_never_splits_a_word() -> None:
    long_text = "supercalifragilistic " * 400

    chunks = arbiter.synthesize(long_text, "whatsapp").channel_payload["chunks"]

    assert all("supercalifragilist" not in chunk[-20:] or chunk.endswith("supercalifragilistic") for chunk in chunks)


def test_instagram_is_plain_and_within_a_thousand_characters() -> None:
    text = "**Bold** and `code` and " + ("word " * 800)

    verdict = arbiter.synthesize(text, "instagram")

    assert len(verdict.formatted_text) <= arbiter.INSTAGRAM_LIMIT
    assert "**" not in verdict.formatted_text
    assert "`" not in verdict.formatted_text
    assert verdict.formatted_text.endswith("…")


def test_telegram_produces_an_inline_keyboard() -> None:
    verdict = arbiter.synthesize("You have three branches.", "telegram")
    payload = verdict.channel_payload

    assert payload["parse_mode"] == "MarkdownV2"
    keyboard = payload["reply_markup"]["inline_keyboard"]
    assert keyboard[0][0]["callback_data"] == "diff"


def test_the_app_gets_full_markdown_and_chips() -> None:
    verdict = arbiter.synthesize("| a | b |\n|---|---|\n| 1 | 2 |", "app")
    payload = verdict.channel_payload

    assert payload["format"] == "markdown"
    assert "|" in verdict.formatted_text, "the app renders tables"
    assert isinstance(payload["action_chips"], list)


def test_an_unknown_channel_falls_back_to_the_app_format() -> None:
    verdict = arbiter.synthesize("hello", "carrier-pigeon")

    assert verdict.channel_payload["format"] == "markdown"


# ---------------------------------------------------------------- 4. speed


def test_the_synthesis_costs_under_a_twentieth_of_a_millisecond() -> None:
    fastest = _fastest_call_microseconds()

    assert fastest < 50, f"fastest call {fastest:.2f} us"


def _fastest_call_microseconds() -> float:
    best = float("inf")

    for _ in range(300):
        started = time.thread_time()
        arbiter.synthesize("You have three branches.", "whatsapp")
        best = min(best, (time.thread_time() - started) * 1_000_000)

    return best


# ---------------------------------------------------------------- 5. the endpoint


def test_the_endpoint_ships_the_cleaned_reply(monkeypatch) -> None:
    import base64
    import hashlib
    import hmac

    from fastapi.testclient import TestClient

    from app.main import app

    secret = "x" * 32
    monkeypatch.setenv("INDOONE_AUTH_SECRET", secret)
    monkeypatch.setenv("INDOONE_MODEL_ARBITER_ENABLED", "true")

    async def fake_reply(message: str, history=None, document_context="", extra_system_instruction="") -> str:
        return "You are a helpful AI assistant.\n\nDocker is a container runtime."

    monkeypatch.setattr("app.api.chat.chat.generate_reply", fake_reply)

    def encode(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    user = encode(b"egress-user")
    stamp = encode(str(int(time.time())).encode("ascii"))
    signature = encode(hmac.new(secret.encode("ascii"), f"{user}.{stamp}".encode("ascii"), hashlib.sha256).digest())

    response = TestClient(app).post(
        "/api/chat",
        json={"message": "what is docker"},
        headers={"Authorization": f"Bearer {user}.{stamp}.{signature}"},
    )

    assert response.status_code == 200
    assert "helpful AI assistant" not in response.json()["reply"]
    assert "Docker" in response.json()["reply"]
