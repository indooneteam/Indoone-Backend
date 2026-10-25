"""The one place the schema is created and evolved.

Before this module, every store carried its own ``CREATE TABLE IF NOT EXISTS`` and its
own ad-hoc surgery - a column added if it was missing, a table rebuilt to gain a
foreign key. That works while a schema only ever grows, and stops working the moment a
column must be renamed, a type must change, or two deployments must agree on what the
schema is. There was no version anywhere, so nothing could answer "is this database
current", and nothing could be rolled back.

Now the schema is Alembic's: one baseline revision holding every table and index as
they stood, and one file per change after it. The stores call ``ensure_schema()`` and
own no DDL at all.

Why the application still runs the upgrade itself
-------------------------------------------------
A deploy that forgets to run ``alembic upgrade head`` should fail loudly, not serve a
half-built database, so the default is to migrate on start. On more than one instance
that is a race, so ``INDOONE_MIGRATE_ON_START=0`` turns it off and the deploy runs the
CLI once, before the service starts.

The baseline is idempotent (every statement is ``IF NOT EXISTS``), which is what makes
adopting an existing deployment safe: the first run creates whatever is missing,
records the revision, and leaves the data alone.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import closing
from pathlib import Path

from app.storage import database

logger = logging.getLogger("indoone.migrations")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _PROJECT_ROOT / "alembic.ini"

# One upgrade per database per process. Without this, every store that initialises and
# every worker that boots at the same moment would run the migration, and the second
# one would find the version table mid-write. Keyed by URL rather than a single flag,
# because a test suite points the app at a fresh file per test and each of those needs
# its own migration.
_UPGRADE_LOCK = threading.Lock()
_UPGRADED: set[str] = set()

# The revision that describes the schema as it was when migrations were introduced.
BASELINE_REVISION = "0001_baseline"


def migrate_on_start() -> bool:
    """Whether this process should bring the schema up to date itself.

    On unless explicitly switched off. A deployment that never set the variable gets
    a working database rather than a stack trace on the first query.
    """

    raw = os.getenv("INDOONE_MIGRATE_ON_START", "").strip().casefold()
    return raw not in {"0", "false", "no", "off"}


def _alembic_config():
    """The Alembic config, pointed at this project's ini file."""

    from alembic.config import Config

    if not _ALEMBIC_INI.is_file():
        raise RuntimeError(f"alembic.ini not found at {_ALEMBIC_INI}")
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_PROJECT_ROOT / "migrations"))
    return config


def current_revision(url: str | None = None) -> str | None:
    """The revision the database is on, or ``None`` when it has never been migrated.

    Read straight from the version table rather than through Alembic, because this is
    called on the health path and must be cheap and must not raise.
    """

    try:
        with closing(database.connect(url=url)) as connection:
            if not connection.table_exists("alembic_version"):
                return None
            row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return str(row[0])


def upgrade_to_head(url: str | None = None) -> str | None:
    """Apply every pending revision. Returns the revision the database ends on.

    ``url`` is for a caller that does not use the application default - the
    conversation store keeps its own SQLite file, and it must migrate that file.
    """

    from alembic import command

    config = _alembic_config()
    config.attributes["database_url"] = url or database.database_url()
    command.upgrade(config, "head")
    return current_revision(url)


def ensure_schema(url: str | None = None) -> str | None:
    """Bring the schema up to date, once per process.

    Safe to call from every store's ``initialize()``: the first caller does the work
    and the rest return immediately. When ``INDOONE_MIGRATE_ON_START=0`` it only
    reports the revision, so a service that must not migrate can still start.
    """

    resolved = url or database.database_url()
    with _UPGRADE_LOCK:
        if resolved in _UPGRADED:
            return current_revision(resolved)
        if not migrate_on_start():
            revision = current_revision(resolved)
            logger.info("schema migrations are handled by the deploy; on revision %s", revision)
            return revision
        revision = upgrade_to_head(resolved)
        _UPGRADED.add(resolved)
        logger.info("schema is on revision %s", revision)
        return revision


def reset_for_tests() -> None:
    """Forget that this process migrated. For tests that switch database files."""

    with _UPGRADE_LOCK:
        _UPGRADED.clear()
