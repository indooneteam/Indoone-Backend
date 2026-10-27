"""Markers this suite uses.

`postgres` marks a test that needs a real Postgres. Those tests are not broken and they
are not unimplemented: they assert things only Postgres can answer - `dialect()` is
"postgres", blobs are BYTEA, `for_update()` emits FOR UPDATE - so pointing them at SQLite
would test the wrong database and pass for the wrong reason.

They run in CI, in the `postgres-store` job, against a postgres:16 service container. This
marker exists so they can be selected on their own:

    pytest -m postgres      # the Postgres-only tests
    pytest -m "not postgres"  # everything that runs anywhere
"""

from __future__ import annotations


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "postgres: needs a real Postgres; set INDOONE_TEST_DATABASE_URL, or let the "
        "postgres-store CI job supply one",
    )
