"""Agent 08 - the Context Condenser.

The property: **send the context the turn needs, and no more** - and when in doubt, send it.
A chat product that guesses "standalone" wrongly answers a question the user did not ask,
which is worse than the tokens it saved.
"""

from __future__ import annotations

import time

import pytest

from app.agents import load

condenser = load("08_context_condenser")

HISTORY = [
    {"role": "user", "content": "list branches in indoone"},
    {"role": "assistant", "content": "branch feat/auth, commit 5b287cb"},
    {"role": "user", "content": "open pull request 12"},
    {"role": "assistant", "content": "PR 12 is about expiring tokens"},
    {"role": "user", "content": "what changed in it"},
    {"role": "assistant", "content": "two files, both in the auth path"},
]


# ---------------------------------------------------------------- 1. zero history


@pytest.mark.parametrize(
    "prompt",
    [
        "what is docker",
        "explain how transactions work in postgres",
        "kya hai postgres",
        "ಏನು ಡಾಕರ್",
    ],
)
def test_a_self_contained_question_is_sent_alone(prompt: str) -> None:
    verdict = condenser.condense(prompt, HISTORY)

    assert verdict.requires_history is False
    assert verdict.condensed_messages == []
    assert verdict.dependency_type == condenser.INDEPENDENT
    assert verdict.tokens_saved_estimate > 0


def test_a_fast_path_turn_needs_no_history() -> None:
    """Agent 05's verdict is the strongest evidence a turn stands on its own."""
    class Intent:
        is_fast_path = True
        resource = "HOLIDAY_CALENDAR"

    verdict = condenser.condense("naale raja na?", HISTORY, intent_metadata=Intent())

    assert verdict.requires_history is False


def test_a_greeting_is_not_a_follow_up() -> None:
    assert condenser.condense("hi", HISTORY).requires_history is False


def test_no_history_means_no_history() -> None:
    verdict = condenser.condense("show its diff", [])

    assert verdict.requires_history is False
    assert verdict.condensed_messages == []


# ---------------------------------------------------------------- 2. anaphora


@pytest.mark.parametrize(
    ("prompt", "dependency"),
    [
        ("adara diff thorisu", "ANAPHORA_RESOLVED"),
        ("show its logs", "ANAPHORA_RESOLVED"),
        ("uska bug fix karo", "ANAPHORA_RESOLVED"),
        ("daani chupinchu", "ANAPHORA_RESOLVED"),
        ("ಅದರಲ್ಲಿ ಬಗ್ ಎಲ್ಲಿದೆ?", "ANAPHORA_RESOLVED"),
    ],
)
def test_a_pronoun_keeps_the_context(prompt: str, dependency: str) -> None:
    verdict = condenser.condense(prompt, HISTORY)

    assert verdict.requires_history is True
    assert verdict.dependency_type == dependency
    assert verdict.resolved_target is not None


def test_the_pronoun_is_bound_to_the_nearest_entity() -> None:
    verdict = condenser.condense("show its diff", HISTORY)

    # the most recent mention wins: PR 12, not the branch from two turns earlier
    assert verdict.resolved_target == "pull_request:12"


def test_a_language_shift_keeps_the_context() -> None:
    """Turn one is English, turn two is Kannada - the thread does not restart."""
    verdict = condenser.condense("ಅದರಲ್ಲಿ ಬಗ್ ಎಲ್ಲಿದೆ?", HISTORY)

    assert verdict.requires_history is True
    assert verdict.resolved_target is not None


@pytest.mark.parametrize(
    "prompt",
    ["what did I say?", "what did we discuss", "summary so far", "ಹಿಂದೆ ಏನು ಹೇಳಿದೆ"],
)
def test_a_question_about_the_conversation_keeps_it(prompt: str) -> None:
    """No pronoun, but the conversation is the subject."""
    assert condenser.condense(prompt, HISTORY).requires_history is True


def test_an_unclear_turn_keeps_its_context() -> None:
    """When in doubt, send the context: guessing wrong costs the user an answer."""
    assert condenser.condense("list branches", HISTORY).requires_history is True


# ---------------------------------------------------------------- 3. compression


def test_a_long_conversation_becomes_two_turns_and_one_line() -> None:
    long_history = []
    for index in range(10):
        long_history.append({"role": "user", "content": f"question {index} about branch feat/auth"})
        long_history.append({"role": "assistant", "content": f"answer {index} about commit 5b287cb"})

    verdict = condenser.condense("show its diff", long_history)

    assert verdict.requires_history is True
    assert len(verdict.condensed_messages) == condenser.FULL_TURNS * 2 + 1
    assert verdict.condensed_messages[0]["role"] == "system"
    assert verdict.condensed_messages[0]["content"].startswith("[Earlier:")
    assert verdict.tokens_saved_estimate > 0


def test_greetings_do_not_survive_into_the_digest() -> None:
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "work on branch feat/login"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "sure"},
    ]

    verdict = condenser.condense("show its diff", history)
    digest = verdict.condensed_messages[0]["content"]

    assert "hello" not in digest.lower()


# ---------------------------------------------------------------- 4. speed


def test_the_decision_costs_under_a_twentieth_of_a_millisecond() -> None:
    best = min(_mean_microseconds("show its diff") for _ in range(3))

    assert best < condenser.PROBE_BUDGET_MILLISECONDS * 1000, f"{best:.4f} us"


def _mean_microseconds(prompt: str) -> float:
    started = time.thread_time()
    for _ in range(500):
        condenser.condense(prompt, HISTORY)
    return (time.thread_time() - started) * 1_000_000 / 500
