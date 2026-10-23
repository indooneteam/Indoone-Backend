from app.ai.memory import extract_memory_candidates, resolve_memory_update
from app.ai.orchestrator import execute_plan, plan_request


def test_memory_extraction_requires_explicit_user_statement() -> None:
    candidates = extract_memory_candidates("Call me Bro from now on.")
    assert len(candidates) == 1
    assert candidates[0].key == "nickname"
    assert candidates[0].value == "Bro from now on"
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


def test_memory_candidates_endpoint_accepts_a_message(monkeypatch) -> None:
    """The endpoint used to name a request model that was never imported.

    That is not only a broken doc: FastAPI validates the body against the annotation, so
    the route could not accept a request at all. This checks the route answers, which is
    what a real caller does.
    """
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setenv("INDOONE_AUTH_SECRET", "x" * 32)
    with TestClient(main.app) as client:
        response = client.post("/api/ai/memory/candidates", json={"message": "Call me Bro from now on."})

    assert response.status_code == 200, response.text
    assert "candidates" in response.json()


def test_the_openapi_schema_builds() -> None:
    """The API docs are generated from this schema, so it failing means /docs is dead.

    A single unresolvable annotation is enough to bring the whole thing down, which is
    what happened: one route named a model that was never imported, and every documented
    path went with it.
    """
    import app.main as main

    # No warning filter here on purpose. A path declared in two routers makes FastAPI warn
    # about a duplicate operation ID, and CI runs pytest with -W error, so this test fails
    # if a duplicate is ever introduced again. That is how the four duplicates were found.
    schema = main.app.openapi()

    assert schema["paths"], "the schema built but documents no paths"
    assert "/api/chat" in schema["paths"]

