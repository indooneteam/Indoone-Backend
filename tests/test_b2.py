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


def test_b2_object_key_validation_rejects_unsafe_values() -> None:
    for value in ("", "/absolute/path", "models/../secret", "models//file", "models/./file", "models/file\n"):
        with pytest.raises(B2StorageError, match="invalid B2 object key"):
            B2Storage._validate_object_key(value)


def test_b2_object_key_validation_normalizes_backslashes() -> None:
    assert B2Storage._validate_object_key("models\\indoone\\weights.bin") == "models/indoone/weights.bin"


def test_b2_upload_rejects_oversized_local_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    local_path = tmp_path / "large.bin"
    local_path.write_bytes(b"12345")
    monkeypatch.setenv("B2_MAX_UPLOAD_BYTES", "4")

    class FakeClient:
        def put_object(self, **kwargs):
            raise AssertionError("oversized upload must be rejected before upload")

    storage = object.__new__(B2Storage)
    storage.bucket_name = "bucket"
    storage._client = FakeClient()

    with pytest.raises(B2StorageError, match="exceeds configured size limit"):
        storage.upload_file(local_path, "models/test.bin")


def test_b2_download_rejects_oversized_content_length(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("B2_MAX_DOWNLOAD_BYTES", "4")

    class FakeResponse:
        headers = {"Content-Length": "5"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            raise AssertionError("body should not be read after size rejection")

    storage = object.__new__(B2Storage)
    storage._download_url = "https://download.example"
    storage._auth_token = "token"
    storage.bucket_name = "bucket"
    monkeypatch.setattr("app.storage.b2.urlopen", lambda *args, **kwargs: FakeResponse())

    target = tmp_path / "model.bin"
    with pytest.raises(B2StorageError, match="exceeds configured size limit"):
        storage.download_file("models/test.bin", target)
    assert not target.exists()


def test_b2_download_enforces_streaming_limit_without_content_length(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("B2_MAX_DOWNLOAD_BYTES", "4")

    class FakeResponse:
        headers = {}
        chunks = iter((b"1234", b"5"))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size: int) -> bytes:
            return next(self.chunks, b"")

    storage = object.__new__(B2Storage)
    storage._download_url = "https://download.example"
    storage._auth_token = "token"
    storage.bucket_name = "bucket"
    monkeypatch.setattr("app.storage.b2.urlopen", lambda *args, **kwargs: FakeResponse())

    target = tmp_path / "model.bin"
    with pytest.raises(B2StorageError, match="exceeds configured size limit"):
        storage.download_file("models/test.bin", target)
    assert not target.exists()
