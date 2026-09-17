from urllib.parse import parse_qs, urlparse

from app.capabilities.canva import _READ_SCOPE, _WRITE_SCOPE, build_canva_authorization


def test_canva_authorization_uses_pkce(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CANVA_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_CANVA_CLIENT_SECRET", "client-secret")
    payload = build_canva_authorization(
        "state-1234567890123456",
        "https://example.com/callback",
        include_write=True,
    )
    params = parse_qs(urlparse(str(payload["authorization_url"])).query)
    assert params["client_id"] == ["client-id"]
    assert params["state"] == ["state-1234567890123456"]
    assert params["code_challenge_method"] == ["S256"]
    assert params["code_challenge"]
    assert params["scope"] == [f"{_READ_SCOPE} {_WRITE_SCOPE}"]
    assert payload["pkce_code_verifier"]


def test_canva_authorization_read_only(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_CANVA_CLIENT_ID", "client-id")
    monkeypatch.setenv("INDOONE_CANVA_CLIENT_SECRET", "client-secret")
    payload = build_canva_authorization(
        "state-1234567890123456",
        "https://example.com/callback",
        include_write=False,
    )
    params = parse_qs(urlparse(str(payload["authorization_url"])).query)
    assert params["scope"] == [_READ_SCOPE]
    assert _WRITE_SCOPE not in params["scope"][0]
