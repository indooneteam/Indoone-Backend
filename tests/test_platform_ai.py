from app.ai.memory import extract_memory_candidates, resolve_memory_update
from app.ai.orchestrator import execute_plan, plan_request


def test_memory_extraction_requires_explicit_user_statement() -> None:
    candidates = extract_memory_candidates("Call me Bro from now on.")
    assert len(candidates) == 1
    assert candidates[0].key == "nickname"
    assert candidates[0].value == "bro from now on"
    assert candidates[0].confidence >= 0.9


def test_memory_update_replaces_mutable_value() -> None:
    candidates = extract_memory_candidates("I prefer concise answers.")
    updated = resolve_memory_update({"preference": "long answers"}, candidates)
    assert updated["preference"] == "concise answers"


def test_calculation_plan_uses_local_tool() -> None:
    plan = plan_request("calculate 12 * 8")
    assert plan.intent.name == "calculation"
    result = execute_plan(plan)
    assert result is not None
    assert result.safe is True
    assert result.output == "96"
