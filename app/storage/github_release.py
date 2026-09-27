"""Private GitHub Release asset storage helpers for Indoone model artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import httpx


class GitHubReleaseStorageError(RuntimeError):
    """Raised when a GitHub Release model download fails."""


class GitHubReleaseStorage:
    """Download model artifacts from a private GitHub Release."""

    API_ROOT = "https://api.github.com"
    MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024

    def __init__(self) -> None:
        self.repository = os.getenv("GITHUB_MODEL_REPOSITORY", "").strip().strip("/")
        self.release_tag = os.getenv("GITHUB_MODEL_RELEASE_TAG", "").strip()
        self.token = os.getenv("GITHUB_MODEL_TOKEN", "").strip()

        missing = [
            name
            for name, value in (
                ("GITHUB_MODEL_REPOSITORY", self.repository),
                ("GITHUB_MODEL_RELEASE_TAG", self.release_tag),
                ("GITHUB_MODEL_TOKEN", self.token),
            )
            if not value
        ]
        if missing:
            raise GitHubReleaseStorageError(
                f"Missing GitHub model configuration: {', '.join(missing)}"
            )

    @classmethod
    def configured(cls) -> bool:
        return all(
            os.getenv(name, "").strip()
            for name in (
                "GITHUB_MODEL_REPOSITORY",
                "GITHUB_MODEL_RELEASE_TAG",
                "GITHUB_MODEL_TOKEN",
            )
        )

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        return {
            "Accept": accept,
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "Indoone-Backend",
        }

    def _release(self) -> dict[str, object]:
        url = (
            f"{self.API_ROOT}/repos/{self.repository}/releases/tags/"
            f"{self.release_tag}"
        )
        try:
            response = httpx.get(
                url,
                headers=self._headers(),
                timeout=30.0,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            raise GitHubReleaseStorageError(
                "GitHub release lookup failed"
            ) from exc

        if response.status_code != 200:
            raise GitHubReleaseStorageError(
                f"GitHub release lookup failed: HTTP {response.status_code}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubReleaseStorageError(
                "GitHub release lookup returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise GitHubReleaseStorageError(
                "GitHub release lookup returned an invalid payload"
            )
        return payload

    @staticmethod
    def _asset_id(release: dict[str, object], asset_name: str) -> int:
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise GitHubReleaseStorageError("GitHub release contains no asset list")

        for asset in assets:
            if not isinstance(asset, dict):
                continue
            if str(asset.get("name", "")) != asset_name:
                continue
            try:
                return int(asset["id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GitHubReleaseStorageError(
                    f"GitHub release asset has an invalid id: {asset_name}"
                ) from exc

        raise GitHubReleaseStorageError(
            f"GitHub release asset not found: {asset_name}"
        )

    def download_file(self, asset_name: str, local_path: Path) -> bool:
        release = self._release()
        asset_id = self._asset_id(release, asset_name)
        url = (
            f"{self.API_ROOT}/repos/{self.repository}/releases/assets/{asset_id}"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = local_path.with_suffix(local_path.suffix + ".download")

        try:
            with httpx.stream(
                "GET",
                url,
                headers=self._headers("application/octet-stream"),
                timeout=httpx.Timeout(60.0, read=120.0, write=60.0, connect=30.0),
                follow_redirects=True,
            ) as response:
                if response.status_code != 200:
                    raise GitHubReleaseStorageError(
                        f"GitHub model download failed: HTTP {response.status_code}"
                    )

                declared = response.headers.get("Content-Length")
                if declared:
                    try:
                        declared_size = int(declared)
                    except ValueError as exc:
                        raise GitHubReleaseStorageError(
                            "GitHub model download returned invalid content length"
                        ) from exc
                    if declared_size < 0 or declared_size > self.MAX_DOWNLOAD_BYTES:
                        raise GitHubReleaseStorageError(
                            "GitHub model download exceeds 2 GiB limit"
                        )

                downloaded = 0
                with temp_path.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > self.MAX_DOWNLOAD_BYTES:
                            raise GitHubReleaseStorageError(
                                "GitHub model download exceeds 2 GiB limit"
                            )
                        handle.write(chunk)

            temp_path.replace(local_path)
            return True
        except GitHubReleaseStorageError:
            temp_path.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError) as exc:
            temp_path.unlink(missing_ok=True)
            raise GitHubReleaseStorageError(
                f"GitHub model download failed for {asset_name}"
            ) from exc


def get_github_release_storage() -> GitHubReleaseStorage | None:
    if not GitHubReleaseStorage.configured():
        return None
    return GitHubReleaseStorage()
