"""Agent 25 - the token cost ledger.

The number this produces is the only evidence that four optimisation agents do anything, so
the arithmetic has to be exact rather than approximately right.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.agents import load

ledger = load("25_token_cost_ledger")
Supervisor = ledger.TokenCostSupervisor


@pytest.fixture(autouse=True)
def _fresh():
    Supervisor.reset_state_for_tests()
    yield
    Supervisor.reset_state_for_tests()


# ---------------------------------------------------------------- 1. the rate card


@pytest.mark.parametrize(
    ("model", "input_tokens", "output_tokens", "expected"),
    [
        ("gemini-1.5-flash", 1_000_000, 0, 0.075),
        ("gemini-1.5-flash", 0, 1_000_000, 0.30),
        ("gemini-1.5-flash", 1_000_000, 1_000_000, 0.375),
        ("gemini-1.5-pro", 1_000_000, 0, 1.25),
        ("gemini-1.5-pro", 0, 1_000_000, 5.00),
        ("azure-gpt-4o-mini", 1_000_000, 0, 0.15),
        ("azure-gpt-4o-mini", 0, 1_000_000, 0.60),
    ],
)
def test_the_cost_matches_the_rate_card(model: str, input_tokens: int, output_tokens: int, expected: float) -> None:
    assert Supervisor.calculate_cost(model, input_tokens, output_tokens) == pytest.approx(expected, abs=1e-6)


def test_cached_tokens_are_charged_at_the_cached_rate() -> None:
    assert Supervisor.calculate_cost("gemini-1.5-flash", 0, 0, 1_000_000) == pytest.approx(0.01875, abs=1e-6)


def test_a_dated_model_identifier_still_matches_its_family() -> None:
    dated = Supervisor.calculate_cost("gemini-1.5-flash-002", 1_000_000, 0)

    assert dated == pytest.approx(0.075, abs=1e-6)


def test_an_unknown_model_falls_back_to_the_highest_known_rate() -> None:
    """Over-reporting is recoverable; under-reporting is a surprise invoice."""
    unknown = Supervisor.calculate_cost("some-new-model", 1_000_000, 0)

    assert unknown == pytest.approx(1.25, abs=1e-6)


def test_nonsense_token_counts_cost_nothing() -> None:
    assert Supervisor.calculate_cost("gemini-1.5-flash", "many", 0) == 0.0  # type: ignore[arg-type]


def test_a_small_call_costs_its_exact_fraction() -> None:
    # 1,000 input + 500 output on Flash.
    expected = (1000 / 1_000_000) * 0.075 + (500 / 1_000_000) * 0.30

    assert Supervisor.calculate_cost("gemini-1.5-flash", 1000, 500) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------- 2. recording


def test_usage_is_partitioned_by_channel() -> None:
    for channel in ("whatsapp", "telegram", "instagram", "app"):
        Supervisor.record_usage(channel, "gemini-1.5-flash", 1000, 500)

    summary = Supervisor.get_summary()

    assert set(summary.by_channel_spend) == {"whatsapp", "telegram", "instagram", "app"}
    for spend in summary.by_channel_spend.values():
        assert spend == pytest.approx(0.000225, abs=1e-6)


def test_usage_is_partitioned_by_model() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 0)
    Supervisor.record_usage("app", "gemini-1.5-pro", 1000, 0)

    summary = Supervisor.get_summary()

    assert set(summary.by_model_spend) == {"gemini-1.5-flash", "gemini-1.5-pro"}
    assert summary.by_model_spend["gemini-1.5-pro"] > summary.by_model_spend["gemini-1.5-flash"]


def test_the_token_totals_add_up() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 100, 50, 20)
    Supervisor.record_usage("app", "gemini-1.5-flash", 200, 60, 10)

    summary = Supervisor.get_summary()

    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 110
    assert summary.total_cached_tokens == 30


def test_a_record_carries_its_own_cost() -> None:
    record = Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500)

    assert record.cost_usd == pytest.approx(0.000225, abs=1e-6)
    assert record.timestamp_epoch > 0


# ---------------------------------------------------------------- 3. savings


@pytest.mark.parametrize(
    "mechanism",
    ["FAST_PATH", "FAST_PATH_INTENT", "CACHE_HIT", "TREASURY_CACHE_HIT", "CONDENSER", "CONTEXT_COMPRESSION", "STREAM_ABORT"],
)
def test_every_mechanism_name_is_understood(mechanism: str) -> None:
    record = Supervisor.record_savings(mechanism, "app", 1000)

    assert record.mechanism in {"FAST_PATH", "CACHE_HIT", "CONDENSER", "STREAM_ABORT"}
    assert record.cost_saved_usd > 0


def test_savings_are_attributed_by_mechanism() -> None:
    Supervisor.record_savings("FAST_PATH", "whatsapp", 5000)
    Supervisor.record_savings("CACHE_HIT", "telegram", 3000)

    summary = Supervisor.get_summary()

    assert set(summary.by_mechanism_savings) == {"FAST_PATH", "CACHE_HIT"}
    assert summary.total_tokens_saved == 8000


def test_savings_are_costed_against_the_reference_model() -> None:
    flash = Supervisor.record_savings("CACHE_HIT", "app", 1_000_000, "gemini-1.5-flash")
    pro = Supervisor.record_savings("CACHE_HIT", "app", 1_000_000, "gemini-1.5-pro")

    assert flash.cost_saved_usd == pytest.approx(0.30, abs=1e-6)
    assert pro.cost_saved_usd == pytest.approx(5.00, abs=1e-6)


# ---------------------------------------------------------------- 4. the ratio


def test_efficiency_is_the_saved_share_of_the_hypothetical_bill() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 0, 1_000_000)      # $0.30 spent
    Supervisor.record_savings("CACHE_HIT", "app", 1_000_000, "gemini-1.5-flash")  # $0.30 saved

    summary = Supervisor.get_summary()

    assert summary.total_spent_usd == pytest.approx(0.30, abs=1e-6)
    assert summary.total_saved_usd == pytest.approx(0.30, abs=1e-6)
    assert summary.efficiency_ratio == pytest.approx(0.5, abs=0.001)


def test_an_empty_ledger_has_no_ratio_rather_than_a_division() -> None:
    summary = Supervisor.get_summary()

    assert summary.total_spent_usd == 0.0
    assert summary.efficiency_ratio == 0.0


def test_a_system_that_saves_most_of_its_bill_reads_high() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 0, 100_000)          # $0.03
    Supervisor.record_savings("FAST_PATH", "app", 900_000, "gemini-1.5-flash")  # $0.27

    assert Supervisor.get_summary().efficiency_ratio == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------- 5. windows


def test_a_time_window_isolates_its_records() -> None:
    now = time.time()
    Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500, now=now - 1000)
    Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500, now=now)

    all_time = Supervisor.get_summary()
    recent = Supervisor.get_summary(since_epoch=now - 10)

    assert all_time.total_input_tokens == 2000
    assert recent.total_input_tokens == 1000


def test_a_window_with_nothing_in_it_is_empty() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500)

    summary = Supervisor.get_summary(since_epoch=time.time() + 1000)

    assert summary.total_spent_usd == 0.0


# ---------------------------------------------------------------- 6. concurrency and speed


def test_concurrent_records_keep_the_arithmetic_exact() -> None:
    def worker() -> None:
        for _ in range(200):
            Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500)

    threads = [threading.Thread(target=worker) for _ in range(4)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    summary = Supervisor.get_summary()

    assert summary.total_input_tokens == 800_000
    assert summary.total_output_tokens == 400_000
    assert summary.total_spent_usd == pytest.approx(800 * 0.000225, abs=1e-6)


def test_costing_stays_under_its_budget() -> None:
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        Supervisor.calculate_cost("gemini-1.5-flash", 1000, 500, 100)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


def test_recording_stays_under_its_budget() -> None:
    fastest = float("inf")

    for _ in range(300):
        started = time.thread_time()
        Supervisor.record_usage("app", "gemini-1.5-flash", 1000, 500)
        fastest = min(fastest, (time.thread_time() - started) * 1000)

    assert fastest < 0.05, f"fastest {fastest:.4f} ms"


# ---------------------------------------------------------------- 7. chips


def test_the_chip_points_at_the_savings_report_when_savings_lead() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 0, 1000)
    Supervisor.record_savings("CACHE_HIT", "app", 100_000)

    assert [chip["action"] for chip in ledger.action_chips(Supervisor.get_summary())] == ["view_savings_report"]


def test_the_chip_points_at_spend_when_there_is_little_saved() -> None:
    Supervisor.record_usage("app", "gemini-1.5-flash", 0, 1_000_000)

    assert [chip["action"] for chip in ledger.action_chips(Supervisor.get_summary())] == ["view_token_spend"]
