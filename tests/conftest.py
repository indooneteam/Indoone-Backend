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

import os

# The suite calls /api without a token, exactly as a local checkout does. Auth is required
# by default in the application now, so it is switched off here once, for the whole run,
# rather than in every test that happens to hit an endpoint.
os.environ.setdefault("INDOONE_AUTH_REQUIRED", "false")
# The suite makes hundreds of requests from one client address, which the API burst limit
# would read as an attack. It is switched off for the run, exactly as auth is; the tests
# that exercise it turn it back on and clear the buckets first.
os.environ.setdefault("INDOONE_API_LIMITS_ENABLED", "false")
# The ingress defender refuses on cadence, which would make a test depend on how many
# requests it happened to make. Its own suite turns it back on.
os.environ.setdefault("INDOONE_GATEKEEPER_ENABLED", "false")
# The intent reader refuses on phrasing; its own suite turns it back on.
os.environ.setdefault("INDOONE_SENTINEL_ENABLED", "false")
# The privacy guardian rewrites prompts; its own suite turns it back on.
os.environ.setdefault("INDOONE_VAULT_GUARDIAN_ENABLED", "false")
# The language marshal is metadata only; its own suite turns it back on.
os.environ.setdefault("INDOONE_POLYGLOT_MARSHAL_ENABLED", "false")
# The intent extractor is metadata plus a log line; its own suite turns it back on.
os.environ.setdefault("INDOONE_INTENT_EXTRACTOR_ENABLED", "false")
# The treasury serves stored answers; its own suite turns it back on and empties it.
os.environ.setdefault("INDOONE_TREASURY_CACHE_ENABLED", "false")
# The condenser changes what history the model is given; its own suite turns it back on.
os.environ.setdefault("INDOONE_CONTEXT_CONDENSER_ENABLED", "false")
# The arbiter refuses turns and keeps a ledger; its own suite turns it back on.
os.environ.setdefault("INDOONE_TOKEN_ARBITER_ENABLED", "false")
