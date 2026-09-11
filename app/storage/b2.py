"""Backblaze B2 object-storage helpers.

The service uses Backblaze's native B2 API for model artifact downloads and
keeps credentials entirely in environment variables.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class B2StorageError(RuntimeError):
    """Raised when a configured B2 operation fails."""


class B2Storage:
    """Small wrapper around Backblaze's native B2 API."""

    AUTH_URL = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"

    def __init__(self) -> None:
        self.key_id = os.getenv("B2_APPLICATION_KEY_ID", "").strip()
        self.application_key = os.getenv("B2_APPLICATION_KEY", "").strip()
        self.bucket_name = os.getenv("B2_BUCKET_NAME", "").strip()
        self.endpoint = os.getenv("B2_ENDPOINT", "").strip().rstrip("/")

        missing = [
            name
            for name, value in (
                ("B2_APPLICATION_KEY_ID", self.key_id),
                ("B2_APPLICATION_KEY", self.application_key),
                ("B2_BUCKET_NAME", self.bucket_name),
            )
            if not value
        ]
        if missing:
            raise B2StorageError(f"Missing B2 configuration: {', '.join(missing)}")

        self._download_url: str | None = None
        self._auth_token: str | None = None

    @classmethod
    def configured(cls) -> bool:
        return all(
            os.getenv(name, "").strip()
            for name in (
                "B2_APPLICATION_KEY_ID",
                "B2_APPLICATION_KEY",
                "B2_BUCKET_NAME",
            )
        )

    def _authorize(self) -> None:
        credentials = base64.b64encode(
            f"{self.key_id}:{self.application_key}".encode("utf-8")
        ).decode("ascii")
        request = Request(
            self.AUTH_URL,
            headers={"Authorization": f"Basic {credentials}"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise B2StorageError("B2 authorization failed") from exc

        self._download_url = str(payload.get("downloadUrl", "")).rstrip("/")
        self._auth_token = str(payload.get("authorizationToken", ""))
        if not self._download_url or not self._auth_token:
            raise B2StorageError("B2 authorization response was incomplete")

    def _ensure_authorized(self) -> None:
        if not self._download_url or not self._auth_token:
            self._authorize()

    def upload_file(self, local_path: Path, object_key: str) -> None:
        # Uploads are handled by the established S3-compatible path elsewhere.
        raise B2StorageError("Direct model uploads are handled outside the runtime")

    def download_file(self, object_key: str, local_path: Path) -> bool:
        self._ensure_authorized()
        assert self._download_url is not None
        assert self._auth_token is not None

        query = urlencode(
            {
                "bucketName": self.bucket_name,
                "fileName": object_key,
            }
        )
        url = f"{self._download_url}/b2api/v2/b2_download_file_by_name?{query}"
        request = Request(url, headers={"Authorization": self._auth_token}, method="GET")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(local_path.suffix + ".download")
        try:
            with urlopen(request, timeout=60) as response, temp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            temp_path.replace(local_path)
            return True
        except HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            if exc.code == 404:
                return False
            raise B2StorageError("B2 model download failed") from exc
        except (URLError, TimeoutError, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise B2StorageError("B2 model download failed") from exc

    def check_access(self) -> bool:
        self._ensure_authorized()
        return True


def get_b2_storage() -> B2Storage | None:
    """Return a configured B2 client, or None when B2 is intentionally disabled."""

    if not B2Storage.configured():
        return None
    return B2Storage()


def ensure_model_artifacts(model_dir: Path) -> None:
    """Download missing model artifacts from B2 when storage is configured."""

    storage = get_b2_storage()
    if storage is None:
        return

    artifacts = {
        "indoone-small.pt": model_dir / "indoone-small.pt",
        "tokenizer.json": model_dir / "tokenizer.json",
    }
    for filename, local_path in artifacts.items():
        if local_path.exists():
            continue
        storage.download_file(f"models/indoone-small/{filename}", local_path)
