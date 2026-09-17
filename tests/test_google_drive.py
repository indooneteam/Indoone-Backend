from __future__ import annotations

import base64

import pytest

from app.capabilities import google_drive


@pytest.mark.asyncio
async def test_list_drive_files_builds_search_request(monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"{}"
        headers = {"content-type": "application/json"}

        def json(self):
            return {"files": [{"id": "1", "name": "notes.txt"}], "nextPageToken": "next"}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            assert url.endswith("/drive/v3/files")
            assert kwargs["params"]["q"] == "trashed = false and name contains 'notes'"
            assert kwargs["params"]["pageSize"] == 25
            assert kwargs["headers"]["Authorization"] == "Bearer access-token"
            return FakeResponse()

    monkeypatch.setattr(google_drive, "_access_token", lambda _: _async_value("access-token"))
    monkeypatch.setattr(google_drive.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    result = await google_drive.list_drive_files("user-1", "notes", page_size=25)
    assert result["files"][0]["name"] == "notes.txt"
    assert result["next_page_token"] == "next"


@pytest.mark.asyncio
async def test_upload_requires_explicit_approval():
    with pytest.raises(PermissionError):
        await google_drive.upload_drive_file("user-1", "a.txt", "text/plain", b"hello", approved=False)


@pytest.mark.asyncio
async def test_upload_builds_multipart_request(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"id": "file-1", "name": "a.txt"}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, **kwargs):
            assert "uploadType=multipart" in url
            assert kwargs["headers"]["Content-Type"].startswith("multipart/related;")
            assert b'"name":"a.txt"' in kwargs["content"]
            assert b"hello" in kwargs["content"]
            return FakeResponse()

    monkeypatch.setattr(google_drive, "_access_token", lambda _: _async_value("access-token"))
    monkeypatch.setattr(google_drive.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    result = await google_drive.upload_drive_file("user-1", "a.txt", "text/plain", b"hello", approved=True)
    assert result["file"]["id"] == "file-1"


@pytest.mark.asyncio
async def test_download_result_is_base64(monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"hello"
        headers = {"content-type": "text/plain"}

        def json(self):
            return {"id": "1", "name": "a.txt", "mimeType": "text/plain"}

        def raise_for_status(self):
            return None

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, **kwargs):
            if url.endswith("/files/1") and kwargs.get("params", {}).get("alt") == "media":
                return FakeResponse()
            response = FakeResponse()
            response.json = lambda: {"id": "1", "name": "a.txt", "mimeType": "text/plain"}
            return response

    monkeypatch.setattr(google_drive, "_access_token", lambda _: _async_value("access-token"))
    monkeypatch.setattr(google_drive.httpx, "AsyncClient", lambda *args, **kwargs: FakeClient())

    result = await google_drive.get_drive_file("user-1", "1", download=True)
    assert base64.b64decode(result["download"]["content_base64"]) == b"hello"


async def _async_value(value):
    return value
