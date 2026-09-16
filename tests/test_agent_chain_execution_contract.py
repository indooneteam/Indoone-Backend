from app.ai.agent import execute_agent


def test_unresolved_chain_step_is_blocked_not_executed() -> None:
    execution = execute_agent("calculate $result9 + 1")
    assert execution.results[0].safe is False
    assert execution.steps == ()
    assert execution.blocked_steps[0].index == 1
    assert execution.blocked_steps[0].tool == "calculator"
