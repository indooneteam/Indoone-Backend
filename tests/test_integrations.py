from fastapi.testclient import TestClient

from app.main import app


def test_integrations_list_is_safe_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/api/integrations")
    assert response.status_code == 200
    items = response.json()["integrations"]
    assert {item["id"] for item in items} >= {"github", "google_calendar", "slack"}
    assert all(item["configured"] is False for item in items)
    assert all("token" not in item and "secret" not in item for item in items)


def test_integration_lookup_and_missing_provider() -> None:
    with TestClient(app) as client:
        response = client.get("/api/integrations/github")
        missing = client.get("/api/integrations/not-configured")
    assert response.status_code == 200
    assert response.json()["integration"]["auth_type"] == "oauth2"
    assert "repo:read" in response.json()["integration"]["permissions"]
    assert missing.status_code == 404
