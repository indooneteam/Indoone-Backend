import pytest

from scripts.run_server import server_settings


def test_production_disables_server_reload(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_SERVER_RELOAD", "true")

    with pytest.raises(RuntimeError, match="INDOONE_SERVER_RELOAD must be false"):
        server_settings()


def test_production_server_settings_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_ENV", "production")
    monkeypatch.setenv("INDOONE_SERVER_WORKERS", "2")
    monkeypatch.setenv("INDOONE_SERVER_LIMIT_CONCURRENCY", "500")
    monkeypatch.setenv("INDOONE_SERVER_TIMEOUT_KEEP_ALIVE", "10")
    monkeypatch.setenv("INDOONE_SERVER_TIMEOUT_GRACEFUL_SHUTDOWN", "45")
    monkeypatch.setenv("PORT", "8080")

    settings = server_settings()

    assert settings["app"] == "app.main:app"
    assert settings["host"] == "0.0.0.0"
    assert settings["port"] == 8080
    assert settings["workers"] == 2
    assert settings["limit_concurrency"] == 500
    assert settings["timeout_keep_alive"] == 10
    assert settings["timeout_graceful_shutdown"] == 45
    assert settings["reload"] is False
    assert settings["server_header"] is False
    assert settings["proxy_headers"] is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("INDOONE_SERVER_WORKERS", "0"),
        ("INDOONE_SERVER_WORKERS", "9"),
        ("INDOONE_SERVER_LIMIT_CONCURRENCY", "0"),
        ("INDOONE_SERVER_TIMEOUT_KEEP_ALIVE", "0"),
        ("INDOONE_SERVER_TIMEOUT_GRACEFUL_SHUTDOWN", "301"),
        ("PORT", "0"),
        ("PORT", "65536"),
    ],
)
def test_runtime_settings_reject_unsafe_bounds(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError):
        server_settings()


def test_reload_and_multiple_workers_are_not_enabled_together(monkeypatch) -> None:
    monkeypatch.setenv("INDOONE_SERVER_RELOAD", "true")
    monkeypatch.setenv("INDOONE_SERVER_WORKERS", "2")

    with pytest.raises(RuntimeError, match="reload"):
        server_settings()
