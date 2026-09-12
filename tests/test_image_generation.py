from __future__ import annotations

import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from app.capabilities.image_generation import generate_image
from app.main import app


@pytest.mark.anyio
async def test_generate_image_normalizes_local_result(monkeypatch) -> None:
    encoded = base64.b64encode(b"fake-png").decode("ascii")

    class MockTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/generate"
            return httpx.Response(200, json={"provider": "local", "model": "test-model", "image_base64": encoded})

    original = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = MockTransport()
        return original(*args, **kwargs)

    monkeypatch.setattr("app.capabilities.image_generation.httpx.AsyncClient", client_factory)
    monkeypatch.setenv("INDOONE_IMAGE_GENERATOR_URL", "http://generator.local/generate")

    result = await generate_image("a blue square", 512, 512)
    assert result.provider == "local"
    assert result.model == "test-model"
    assert result.image_base64 == encoded


def test_image_generation_endpoint_returns_503_without_runtime() -> None:
    with TestClient(app) as client:
        response = client.post("/api/image-generation", json={"prompt": "test", "width": 512, "height": 512})
    assert response.status_code == 503
