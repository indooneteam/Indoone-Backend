"""Backblaze B2 object-storage helpers."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError


class B2StorageError(RuntimeError):
    """Raised when a configured B2 operation fails."""


class B2Storage:
    """Small wrapper around Backblaze's native API and S3-compatible API."""

    AUTH_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"

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
                ("B2_ENDPOINT", self.endpoint),
            )
            if not value
        ]
        if missing:
            raise B2StorageError(f"Missing B2 configuration: {', '.join(missing)}")

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=self._region_from_endpoint(),
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.application_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 4, "mode": "standard"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
        self._download_url: str | None = None
        self._auth_token: str | None = None

    def _region_from_endpoint(self) -> str:
        marker = "https://s3."
        suffix = ".backblazeb2.com"
        if self.endpoint.startswith(marker) and self.endpoint.endswith(suffix):
            return self.endpoint[len(marker) : -len(suffix)]
        return os.getenv("B2_REGION", "us-east-1")

    @classmethod
    def configured(cls) -> bool:
        return all(
            os.getenv(name, "").strip()
            for name in (
                "B2_APPLICATION_KEY_ID",
                "B2_APPLICATION_KEY",
                "B2_BUCKET_NAME",
                "B2_ENDPOINT",
            )
        )

    def _authorize_native(self) -> None:
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
        except HTTPError as exc:
            try:
                error_payload = json.load(exc)
                code = error_payload.get("code", "authorization_error")
                message = error_payload.get("message", "B2 authorization failed")
            except (json.JSONDecodeError, ValueError, OSError):
                code = "authorization_error"
                message = "B2 authorization failed"
            raise B2StorageError(f"B2 authorization failed: {code}: {message}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise B2StorageError("B2 authorization failed") from exc

        storage_api = payload.get("apiInfo", {}).get("storageApi", {})
        self._download_url = str(storage_api.get("downloadUrl", "")).rstrip("/")
        self._auth_token = str(payload.get("authorizationToken", ""))
        if not self._download_url or not self._auth_token:
            raise B2StorageError("B2 authorization response was incomplete")

    def upload_file(self, local_path: Path, object_key: str) -> None:
        try:
            with local_path.open("rb") as handle:
                self._client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=handle,
                    ContentLength=local_path.stat().st_size,
                )
        except (BotoCoreError, ClientError, OSError) as exc:
            raise B2StorageError(f"B2 upload failed for {object_key}") from exc

    def download_file(self, object_key: str, local_path: Path) -> bool:
        if not self._download_url or not self._auth_token:
            self._authorize_native()
        assert self._download_url is not None
        assert self._auth_token is not None

        url = (
            f"{self._download_url}/file/"
            f"{quote(self.bucket_name, safe='')}/"
            f"{quote(object_key, safe='/')}"
        )
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
        self._authorize_native()
        return True


def get_b2_storage() -> B2Storage | None:
    if not B2Storage.configured():
        return None
    return B2Storage()


def ensure_model_artifacts(model_dir: Path) -> None:
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
