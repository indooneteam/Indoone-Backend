from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.api.integrations import router as integrations_router
from app.capabilities.github import list_github_repositories
from app.main import app


# ... existing tests ...

def test_github_repositories_endpoint_validates_unconnected_user(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        response = client.post("/api/integrations/github/repositories", json={"user_id": "user-two"})
    assert response.status_code == 400
    assert "not connected" in response.json()["message"]
