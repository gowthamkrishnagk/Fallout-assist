"""Copy the standalone application's local SQLite state into empty PostgreSQL.

Set DATABASE_URL first. This is an explicit, one-time migration; runtime failover
is intentionally disabled so Jira publication and audit have one source of truth.
"""

from __future__ import annotations

import os
import sqlite3

from falloutassist_next import storage
from falloutassist_next.config import DATABASE_PATH, DEFAULTS


TABLES = (
    "settings", "connection_config", "source_quality", "source_docs", "app_state", "rules",
    "investigations", "publications", "feedback", "audit_events", "leases",
    "scan_state", "uploaded_documents", "integration_secrets",
)


def main() -> None:
    if not os.getenv("DATABASE_URL", "").strip():
        raise RuntimeError("Set DATABASE_URL to the destination PostgreSQL database")
    if not DATABASE_PATH.is_file():
        raise RuntimeError(f"SQLite fallback database does not exist: {DATABASE_PATH}")
    os.environ["DATABASE_BACKEND"] = "postgres"
    storage.initialize()
    source = sqlite3.connect(DATABASE_PATH.as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source_tables = {row["name"] for row in source.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    counts: dict[str, int] = {}
    try:
        with storage.connect() as target:
            for table in TABLES:
                if table != "settings":
                    count = target.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                    if count:
                        raise RuntimeError(f"Destination {table} already contains data; migration requires an empty database")
            target.execute("DELETE FROM settings")
            for table in TABLES:
                rows = (source.execute(f"SELECT * FROM {table}").fetchall()
                        if table in source_tables else [])
                counts[table] = len(rows)
                if not rows:
                    continue
                columns = list(rows[0].keys())
                placeholders = ",".join("?" for _ in columns)
                target.executemany(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                    [tuple(row[column] for column in columns) for row in rows],
                )
            for key, value in DEFAULTS.items():
                target.execute(
                    "INSERT INTO settings(key,value_json,changed_at) VALUES (?,?,?) "
                    "ON CONFLICT(key) DO NOTHING",
                    (key, storage._json(value), storage.utc_now()),
                )
            for table in ("feedback", "audit_events"):
                target.execute(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"GREATEST(COALESCE((SELECT MAX(id) FROM {table}), 0), 1), "
                    f"EXISTS(SELECT 1 FROM {table}))"
                )
    finally:
        source.close()
    print({"destination": "postgres", "copied": counts})


if __name__ == "__main__":
    main()
