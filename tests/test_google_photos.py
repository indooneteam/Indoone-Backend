from app.capabilities.google_photos import _APPEND_SCOPE, _READ_SCOPE, build_google_photos_authorization


def test_google_photos_authorization_read_scope(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    payload = build_google_photos_authorization("state-1234567890123456", "https://example.com/callback", include_upload=False)
    assert payload["integration"] == "google_photos"
    assert payload["scope"] == _READ_SCOPE
    assert _APPEND_SCOPE not in str(payload["authorization_url"])


def test_google_photos_authorization_upload_scope(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_GOOGLE_CLIENT_SECRET", "client-secret")
    payload = build_google_photos_authorization("state-1234567890123456", "https://example.com/callback", include_upload=True)
    assert _READ_SCOPE in str(payload["authorization_url"])
    assert _APPEND_SCOPE in str(payload["authorization_url"])
