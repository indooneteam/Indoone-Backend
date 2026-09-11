from pathlib import Path

import pytest

from app.storage.b2 import B2Storage, B2StorageError, get_b2_storage


def test_b2_is_optional_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
        "B2_BUCKET_NAME",
        "B2_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    assert B2Storage.configured() is False
    assert get_b2_storage() is None


def test_b2_requires_all_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("B2_BUCKET_NAME", "indoone-training")
    monkeypatch.delenv("B2_ENDPOINT", raising=False)

    with pytest.raises(B2StorageError, match="B2_ENDPOINT"):
        B2Storage()


def test_b2_infers_region_from_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeClient:
        pass

    def fake_client(service_name: str, **kwargs: str) -> FakeClient:
        captured["service"] = service_name
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr("app.storage.b2.boto3.client", fake_client)
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "key-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "secret")
    monkeypatch.setenv("B2_BUCKET_NAME", "indoone-training")
    monkeypatch.setenv("B2_ENDPOINT", "https://s3.eu-central-003.backblazeb2.com")

    storage = B2Storage()

    assert isinstance(storage._client, FakeClient)
    assert captured["service"] == "s3"
    assert captured["region_name"] == "eu-central-003"
    assert captured["endpoint_url"] == "https://s3.eu-central-003.backblazeb2.com"
