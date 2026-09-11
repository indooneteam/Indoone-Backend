from pathlib import Path

from app.ai.behavior_eval import load_cases, score_response


def test_load_behavior_cases() -> None:
    cases = load_cases(Path("data/eval/behavior.jsonl"))
    assert len(cases) == 5
    assert cases[0]["id"] == "instruction"
    assert {case["category"] for case in cases} == {
        "instruction_following",
        "honesty",
        "safety",
        "grounding",
        "conversation",
    }


def test_score_response_tracks_topic_coverage() -> None:
    score = score_response("Indoone uses authorized access with permission.", ["permission", "authorized"])
    assert score["response_nonempty"] is True
    assert score["matched_topics"] == ["permission", "authorized"]
    assert score["topic_coverage"] == 1.0
