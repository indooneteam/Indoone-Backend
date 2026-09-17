from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi.testclient import TestClient


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_AUTH_SECRET = "smoke-auth-secret-32-characters-long-0001"
_APPROVAL_SECRET = "smoke-approval-secret-32-characters-0002"
_ORIGIN = "https://trusted.example"


def _token(user_id: str = "smoke-user") -> str:
    encoded_user = base64.urlsafe_b64encode(user_id.encode("utf-8")).decode("ascii").rstrip("=")
    encoded_timestamp = base64.urlsafe_b64encode(str(int(time.time())).encode("ascii")).decode("ascii").rstrip("=")
    payload = f"{encoded_user}.{encoded_timestamp}".encode("ascii")
    signature = hmac.new(_AUTH_SECRET.encode("utf-8"), payload, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    return f"Bearer {encoded_user}.{encoded_timestamp}.{encoded_signature}"


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="indoone-security-smoke-") as temp_dir:
        os.environ.update(
            {
                "INDOONE_ENV": "production",
                "INDOONE_AUTH_REQUIRED": "true",
                "INDOONE_AUTH_SECRET": _AUTH_SECRET,
                "INDOONE_APPROVAL_SECRET": _APPROVAL_SECRET,
                "INDOONE_CONNECTOR_AUTH_REQUIRED": "true",
                "INDOONE_ALLOWED_HOSTS": "testserver",
                "INDOONE_ALLOWED_ORIGINS": _ORIGIN,
                "INDOONE_TRUST_PROXY_HEADERS": "false",
                "INDOONE_MAX_REQUEST_BYTES": "1024",
                "INDOONE_CAPABILITY_DB": str(Path(temp_dir) / "capabilities.db"),
                "PORT": "18080",
                "INDOONE_SERVER_WORKERS": "2",
                "INDOONE_SERVER_LIMIT_CONCURRENCY": "250",
                "INDOONE_SERVER_TIMEOUT_KEEP_ALIVE": "10",
                "INDOONE_SERVER_TIMEOUT_GRACEFUL_SHUTDOWN": "45",
                "INDOONE_SERVER_RELOAD": "false",
            }
        )

        from app.main import app
        from scripts.run_server import server_settings

        settings = server_settings()
        assert settings["app"] == "app.main:app"
        assert settings["port"] == 18080
        assert settings["workers"] == 2
        assert settings["limit_concurrency"] == 250
        assert settings["timeout_keep_alive"] == 10
        assert settings["timeout_graceful_shutdown"] == 45
        assert settings["reload"] is False
        assert settings["server_header"] is False
        assert settings["proxy_headers"] is False

        auth = {"Authorization": _token()}
        with TestClient(app, base_url="https://testserver") as client:
            response = client.get("/health")
            assert response.status_code == 401

            response = client.get("/health", headers=auth)
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Strict-Transport-Security"].startswith("max-age=31536000")
            assert response.headers["X-Request-ID"]

            response = client.get("/ready", headers=auth)
            assert response.status_code == 200
            assert response.json() == {"status": "ready"}

            response = client.get("/health", headers={**auth, "Origin": _ORIGIN})
            assert response.status_code == 200
            assert response.headers["Access-Control-Allow-Origin"] == _ORIGIN

            response = client.get("/health", headers={**auth, "Origin": "https://untrusted.example"})
            assert response.status_code == 200
            assert "Access-Control-Allow-Origin" not in response.headers

            response = client.get(
                "/health",
                headers={**auth, "Host": "evil.example"},
            )
            assert response.status_code == 400

            response = client.post(
                "/health",
                headers=auth,
                content=b"x" * 2048,
            )
            assert response.status_code == 413
            assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"

        with TestClient(app, base_url="http://testserver") as http_client:
            response = http_client.get(
                "/health",
                headers={**auth, "X-Forwarded-Proto": "https"},
            )
            assert response.status_code == 200
            assert "Strict-Transport-Security" not in response.headers


if __name__ == "__main__":
    run_smoke()
    print("production security smoke: PASS")
