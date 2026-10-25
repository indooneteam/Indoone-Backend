"""Alembic environment.

Two things are deliberately different from the default generated file.

**The URL comes from the application.** ``alembic.ini`` has no ``sqlalchemy.url``;
this module asks ``app.storage.database`` for the database the app itself would use.
That is what keeps ``alembic upgrade head`` and the running service on the same
database, whether that is the SQLite file or Postgres.

**Alembic opens its own connection.** The stores in this codebase talk to the database
through ``app.storage.database``, a small wrapper around SQLite and Postgres with no
models. Alembic, though, needs a SQLAlchemy connection - it cannot drive a wrapper
object. So it is given the same URL the application uses and opens its own connection
to it. Same database, one source of truth for which database that is; two ways of
talking to it, which is unavoidable without rewriting the stores onto SQLAlchemy.

The schema itself is written as SQL in each revision, which is what this codebase
already writes everywhere else.
"""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context

from app.storage import database

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

logger = logging.getLogger("indoone.migrations")

# No models: the schema is written as SQL in each revision, so autogenerate has
# nothing to compare against. Revisions are written by hand, which for this schema -
# fourteen tables and their indexes - is clearer than a generated diff anyway.
target_metadata = None


def _database_url() -> str:
    """The database to migrate.

    A caller that migrates a specific database - the conversation store keeps its own
    SQLite file - passes the URL in ``config.attributes``. Otherwise the application's
    own setting decides, so the CLI and the service always agree.
    """

    supplied = context.config.attributes.get("database_url")
    if isinstance(supplied, str) and supplied.strip():
        return supplied.strip()
    return database.database_url()


def run_migrations_offline() -> None:
    """Emit the SQL to stdout instead of running it.

    This is what a DBA runs to review a change before it touches production:
    ``alembic upgrade head --sql``.
    """

    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _sqlalchemy_url(url: str) -> str:
    """The application's URL, in the form SQLAlchemy needs.

    ``postgresql://`` makes SQLAlchemy reach for psycopg2. This deployment carries
    psycopg 3, so the driver is named explicitly; SQLite needs nothing.
    """

    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    if url.lower().startswith("sqlite://"):
        # The application resolves a SQLite URL to a path with its own rules
        # (``database.sqlite_path``), and creates the directory before opening it.
        # Migrations must land on that exact file, so the path is asked for rather
        # than re-derived here, and the directory is created the same way - otherwise
        # Alembic writes one file and the app reads another, or fails to open it at
        # all because the directory does not exist yet.
        location = database.sqlite_path(url)
        location.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{location}"
    return url


def run_migrations_online() -> None:
    """Run the migrations against the configured database."""

    from sqlalchemy import create_engine

    url = _database_url()
    logger.info("migrating %s", "postgres" if database.is_postgres(url) else "sqlite")
    engine = create_engine(_sqlalchemy_url(url))
    try:
        with engine.begin() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                # A migration and its version bump are one unit: a failure half way
                # must leave neither behind.
                transaction_per_migration=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
