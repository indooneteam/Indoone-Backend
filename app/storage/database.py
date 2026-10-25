"""One place where the database is chosen, opened and spoken to.

The service keeps its state in a single SQLite file, which is fine for one box and one
process. The moment there are two of either it is not: two writers on one file
serialise against each other, and a second server cannot see the first server's file.

This module is the seam that lets the state move without the rest of the code knowing.
It reads ``INDOONE_DATABASE_URL``, opens the right driver, and hands back a connection
with a small surface: ``execute``, ``commit``, and a row you read by column name. A
caller writes its SQL once, with ``?`` placeholders, and does not care which database
it landed on.

    sqlite:///data/indoone_capabilities.db     the default, and what runs today
    postgresql://user:pass@host:5432/indoone   Postgres

Nothing outside this module imports ``sqlite3`` or ``psycopg``.

The three things that are genuinely different between the two, and are answered here:

- **Placeholders.** SQLite uses ``?``, Postgres uses ``%s``. ``execute`` translates.
- **Concurrency.** SQLite serialises writers with ``BEGIN IMMEDIATE``; Postgres takes
  row locks instead, so ``begin_write`` starts a transaction and ``for_update`` gives
  the clause a locking read needs.
- **Binary columns and introspection.** ``BLOB`` is ``BYTEA`` in Postgres, and the
  schema introspection SQLite exposes as ``PRAGMA`` lives in ``information_schema``.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_SQLITE_PATH = "data/indoone_capabilities.db"

# SQLite's own integrity error, re-exported so callers can catch one name on both
# databases instead of importing a driver to name an exception.
IntegrityError: type[Exception] = sqlite3.IntegrityError


def database_url(path: str | Path | None = None) -> str:
    """Return the configured database URL, defaulting to the SQLite file in use today."""

    configured = os.getenv("INDOONE_DATABASE_URL", "").strip()
    if configured:
        return configured
    location = str(path) if path is not None else os.getenv(
        "INDOONE_CAPABILITY_DB", DEFAULT_SQLITE_PATH
    ).strip()
    return f"sqlite:///{location}"


def dialect(url: str | None = None) -> str:
    """Return ``sqlite`` or ``postgres`` for a URL."""

    value = (url or database_url()).strip().lower()
    if value.startswith(("postgres://", "postgresql://")):
        return "postgres"
    if value.startswith("sqlite://"):
        return "sqlite"
    raise ValueError(
        "unsupported INDOONE_DATABASE_URL: use sqlite:///path or postgresql://user:pass@host/db"
    )


def is_postgres(url: str | None = None) -> bool:
    return dialect(url) == "postgres"


def sqlite_path(url: str | None = None) -> Path:
    """Return the file behind a SQLite URL."""

    value = (url or database_url()).strip()
    if not value.lower().startswith("sqlite://"):
        raise ValueError("not a sqlite url")
    return Path(value[len("sqlite://"):].lstrip("/") or DEFAULT_SQLITE_PATH)


def blob_type(url: str | None = None) -> str:
    """The column type for binary payloads."""

    return "BYTEA" if is_postgres(url) else "BLOB"


def for_update(url: str | None = None) -> str:
    """The clause a read needs when the row must stay locked until the write lands."""

    return " FOR UPDATE" if is_postgres(url) else ""


class Cursor:
    """A cursor whose rows can be read by column name on either database."""

    def __init__(self, cursor: Any) -> None:
        self._cursor = cursor

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    def fetchmany(self, size: int = 1) -> list[Any]:
        return self._cursor.fetchmany(size)

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cursor)


class Connection:
    """One database connection, with the differences between the two dialects handled.

    Used as a context manager it commits on the way out and rolls back on an exception,
    which is what ``sqlite3.Connection`` does and what the callers already rely on.
    """

    def __init__(self, raw: Any, kind: str) -> None:
        self._raw = raw
        self.kind = kind

    # -- statements ---------------------------------------------------------

    def _prepare(self, sql: str) -> str:
        return sql if self.kind == "sqlite" else sql.replace("?", "%s")

    def execute(self, sql: str, parameters: Any = ()) -> Cursor:
        if self.kind == "sqlite":
            return Cursor(self._raw.execute(sql, parameters))
        return Cursor(self._raw.cursor().execute(self._prepare(sql), parameters))

    def executemany(self, sql: str, rows: Any) -> Cursor:
        if self.kind == "sqlite":
            return Cursor(self._raw.executemany(sql, rows))
        cursor = self._raw.cursor()
        cursor.executemany(self._prepare(sql), rows)
        return Cursor(cursor)

    def executescript(self, script: str) -> None:
        """Run a multi-statement script.

        SQLite has ``executescript``; Postgres accepts several statements in one
        ``execute``, so the same script runs on both.
        """

        if self.kind == "sqlite":
            self._raw.executescript(script)
            return
        with self._raw.cursor() as cursor:
            cursor.execute(script)
        self._raw.commit()

    # -- transactions -------------------------------------------------------

    def begin_write(self) -> None:
        """Take the write lock before reading, so a read-then-write cannot interleave."""

        if self.kind == "sqlite":
            self._raw.execute("BEGIN IMMEDIATE")
            return
        # Postgres is always inside a transaction once a statement has run; a
        # blocking read is what serialises the two writers, and for_update() is how
        # the caller asks for it.
        self._raw.autocommit = False

    def commit(self) -> None:
        self._raw.commit()

    def rollback(self) -> None:
        self._raw.rollback()

    # -- schema -------------------------------------------------------------

    def table_columns(self, table: str) -> set[str]:
        """Return the column names of a table, on either database."""

        if self.kind == "sqlite":
            rows = self._raw.execute(f"PRAGMA table_info({table})").fetchall()
            return {str(row[1]) for row in rows}
        rows = self._raw.cursor().execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = %s AND table_schema = current_schema()",
            (table,),
        ).fetchall()
        return {str(row["column_name"]) for row in rows}

    def foreign_key_columns(self, table: str) -> set[str]:
        """Return the local columns of a table that reference another table."""

        if self.kind == "sqlite":
            rows = self._raw.execute(f"PRAGMA foreign_key_list({table})").fetchall()
            return {str(row[3]) for row in rows}
        rows = self._raw.cursor().execute(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON kcu.constraint_name = tc.constraint_name "
            " AND kcu.table_name = tc.table_name "
            "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s "
            "  AND tc.table_schema = current_schema()",
            (table,),
        ).fetchall()
        return {str(row["column_name"]) for row in rows}

    def integrity_problems(self) -> int:
        """Count broken foreign keys.

        SQLite can be asked to check; Postgres enforces every key as it is written, so
        there is nothing left to find and the answer is always zero.
        """

        if self.kind == "sqlite":
            return len(self._raw.execute("PRAGMA foreign_key_check").fetchall())
        return 0

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._raw.close()

    def __enter__(self) -> "Connection":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self._raw.commit()
        else:
            self._raw.rollback()
        return False


def connect(path: str | Path | None = None, url: str | None = None) -> Connection:
    """Open a connection to the configured database."""

    resolved = url or database_url(path)
    kind = dialect(resolved)
    if kind == "sqlite":
        location = sqlite_path(resolved)
        location.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(str(location))
        raw.row_factory = sqlite3.Row
        return Connection(raw, "sqlite")
    return Connection(_connect_postgres(resolved), "postgres")


def _connect_postgres(url: str) -> Any:
    """Open Postgres, returning rows keyed by column name like SQLite's Row."""

    global IntegrityError
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - only when Postgres is chosen
        raise RuntimeError(
            "Postgres was configured but psycopg is not installed; "
            "install psycopg[binary] or set INDOONE_DATABASE_URL back to sqlite"
        ) from exc
    IntegrityError = psycopg.errors.IntegrityError
    return psycopg.connect(url, row_factory=dict_row)


@contextmanager
def transaction(path: str | Path | None = None) -> Iterator[Connection]:
    """Open a connection, commit on a clean exit, roll back on an exception."""

    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
