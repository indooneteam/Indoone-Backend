from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from app.capabilities.data_analysis import analyze_payload
from app.main import app


def test_capability_registry_exposes_core_features() -> None:
    with TestClient(app) as client:
        response = client.get("/api/capabilities")
    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["capabilities"]}
    assert {"chat", "memory", "projects", "tasks", "data_analysis", "research", "vision", "image_generation"} <= ids


def test_memory_round_trip_search_and_isolation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        created = client.post(
            "/api/memory",
            json={"user_id": "u1", "key": "nickname", "value": "Bro", "confidence": 0.99},
        )
        assert created.status_code == 200
        assert created.json()["memory"]["value"] == "Bro"

        other = client.post(
            "/api/memory",
            json={"user_id": "u2", "key": "nickname", "value": "Other"},
        )
        assert other.status_code == 200

        listed = client.get("/api/memory", params={"user_id": "u1"})
        assert listed.status_code == 200
        assert listed.json()["memories"][0]["key"] == "nickname"

        searched = client.get("/api/memory", params={"user_id": "u1", "q": "Bro"})
        assert searched.status_code == 200
        assert [item["value"] for item in searched.json()["memories"]] == ["Bro"]

        assert client.get("/api/memory", params={"user_id": "u2", "q": "Bro"}).json()["memories"] == []

        updated = client.post(
            "/api/memory",
            json={"user_id": "u1", "key": "nickname", "value": "Boss", "confidence": 1.0},
        )
        assert updated.json()["memory"]["id"] == created.json()["memory"]["id"]
        assert updated.json()["memory"]["created_at"] == created.json()["memory"]["created_at"]


def test_project_and_task_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CAPABILITY_DB", str(tmp_path / "capabilities.db"))
    with TestClient(app) as client:
        project = client.post(
            "/api/projects",
            json={"user_id": "u1", "name": "AI work", "instructions": "Be concise"},
        )
        assert project.status_code == 200
        task = client.post(
            "/api/tasks",
            json={"user_id": "u1", "title": "Daily brief", "prompt": "Summarize", "schedule": "RRULE:FREQ=DAILY"},
        )
        assert task.status_code == 200
        assert client.get("/api/projects", params={"user_id": "u1"}).json()["projects"]
        assert client.get("/api/tasks", params={"user_id": "u1"}).json()["tasks"]


def test_csv_analysis() -> None:
    payload = b"name,score\na,10\nb,20\n"
    result = analyze_payload("scores.csv", payload)
    assert result["rows"] == 2
    assert result["summary"]["score"]["mean"] == 15


def test_media_intake() -> None:
    content = base64.b64encode(b"test-image-bytes").decode("ascii")
    with TestClient(app) as client:
        response = client.post(
            "/api/media",
            json={"filename": "photo.png", "mime_type": "image/png", "content_base64": content},
        )
    assert response.status_code == 200
    assert response.json()["vision_ready"] is True
