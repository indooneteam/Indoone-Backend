"""Backblaze B2 S3-compatible object storage helpers.

The service only uses B2 when all required environment variables are present.
Credentials are read from the environment and are never written to logs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError


_ENDPOINT_PATTERN = re.compile(r"^https://s3\\.(?P<region>[a-z0-9-]+)\\.backblazeb2\\.com/?$")


class B2StorageError(RuntimeError):
    """Raised when a configured B2 operation fails."""


class B2Storage:
    """Small wrapper around Backblaze's S3-compatible API."""

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

        match = _ENDPOINT_PATTERN.match(self.endpoint + "/")
        region = match.group("region") if match else os.getenv("B2_REGION", "us-east-1")
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=region,
            aws_access_key_id=self.key_id,
            aws_secret_access_key=self.application_key,
        )

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

    def upload_file(self, local_path: Path, object_key: str) -> None:
        try:
            self._client.upload_file(str(local_path), self.bucket_name, object_key)
        except (BotoCoreError, ClientError) as exc:
            raise B2StorageError(f"B2 upload failed for {object_key}") from exc

    def download_file(self, object_key: str, local_path: Path) -> bool:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(local_path.suffix + ".download")
        try:
            self._client.download_file(self.bucket_name, object_key, str(temp_path))
            temp_path.replace(local_path)
            return True
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NoSuchBucket"}:
                temp_path.unlink(missing_ok=True)
                return False
            temp_path.unlink(missing_ok=True)
            raise B2StorageError(f"B2 download failed for {object_key}") from exc
        except BotoCoreError as exc:
            temp_path.unlink(missing_ok=True)
            raise B2StorageError(f"B2 download failed for {object_key}") from exc

    def check_access(self) -> bool:
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            return True
        except (BotoCoreError, ClientError) as exc:
            raise B2StorageError("B2 bucket access check failed") from exc


def get_b2_storage() -> B2Storage | None:
    """Return a configured B2 client, or None when B2 is intentionally disabled."""

    if not B2Storage.configured():
        return None
    return B2Storage()
