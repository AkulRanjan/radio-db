"""PostgreSQL catalog for production (``RADIODB_DATABASE_URL=postgresql://...``).

Differences from SQLite: real ``timestamptz``/``bytea``/``jsonb`` types, and full-text search
via a generated ``tsvector`` column with a GIN index (English stemming, ``ts_rank_cd``,
``ts_headline`` snippets). Requires ``pip install radiodb[postgres]`` (psycopg 3 + psycopg-pool).
"""

from __future__ import annotations

import re
import threading
from contextlib import contextmanager
from typing import Any, Iterator

from .base import BaseCatalog, _segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    title        TEXT,
    sha256       TEXT NOT NULL,
    blob_key     TEXT NOT NULL,
    format       TEXT NOT NULL,
    sample_rate  INTEGER NOT NULL,
    channels     INTEGER NOT NULL,
    duration_s   DOUBLE PRECISION NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    expires_at   TIMESTAMPTZ NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    producers    JSONB,
    UNIQUE (user_id, sha256)
);
CREATE INDEX IF NOT EXISTS ix_recordings_user ON recordings(user_id);
CREATE INDEX IF NOT EXISTS ix_recordings_expires ON recordings(expires_at);
CREATE INDEX IF NOT EXISTS ix_recordings_blob ON recordings(blob_key);

CREATE TABLE IF NOT EXISTS speakers (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT,
    enrolled    BOOLEAN NOT NULL DEFAULT FALSE,
    centroid    BYTEA NOT NULL,
    n_samples   INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_speakers_user ON speakers(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_speakers_user_name ON speakers(user_id, lower(name)) WHERE name IS NOT NULL;

CREATE TABLE IF NOT EXISTS segments (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recording_id  TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    speaker_id    TEXT REFERENCES speakers(id) ON DELETE SET NULL,
    start_s       DOUBLE PRECISION NOT NULL,
    end_s         DOUBLE PRECISION NOT NULL,
    text          TEXT NOT NULL DEFAULT '',
    words         JSONB NOT NULL DEFAULT '[]',
    embedding     BYTEA,
    tsv           TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS ix_segments_recording ON segments(recording_id, start_s);
CREATE INDEX IF NOT EXISTS ix_segments_speaker ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS ix_segments_user ON segments(user_id);
CREATE INDEX IF NOT EXISTS ix_segments_tsv ON segments USING GIN (tsv);

CREATE TABLE IF NOT EXISTS segment_layers (
    segment_id  BIGINT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    layer       TEXT NOT NULL,
    producer    TEXT NOT NULL,
    data        JSONB,
    vector      BYTEA,
    PRIMARY KEY (segment_id, layer)
);
CREATE INDEX IF NOT EXISTS ix_segment_layers_layer ON segment_layers(layer);
-- Work queue (schema v2). One active (queued/running) job per recording and kind.
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,                -- process | enrich | purge
    recording_id  TEXT REFERENCES recordings(id) ON DELETE CASCADE,   -- NULL for system jobs
    payload       JSONB NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL,                -- queued | running | done | failed
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    run_after     TIMESTAMPTZ NOT NULL,
    locked_by     TEXT,
    locked_until  TIMESTAMPTZ,
    last_error    TEXT,
    created_at    TIMESTAMPTZ NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL,
    finished_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_jobs_ready ON jobs(status, run_after);
CREATE INDEX IF NOT EXISTS ix_jobs_user ON jobs(user_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_active ON jobs(COALESCE(recording_id, ''), kind)
    WHERE status IN ('queued', 'running');

CREATE TABLE IF NOT EXISTS radiodb_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""

_SCHEMA_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_HEADLINE = "StartSel=[, StopSel=], MaxWords=24, MinWords=8, MaxFragments=1, FragmentDelimiter=..."


class PostgresCatalog(BaseCatalog):
    """Pooled connections, safe to share across API worker threads.

    A transaction pins one pooled connection to the current thread for its duration (nested
    transactions become savepoints); every other statement borrows a connection briefly.
    """

    placeholder = "%s"
    skip_locked = " FOR UPDATE SKIP LOCKED"

    def __init__(self, url: str, schema: str | None = None, min_size: int = 1, max_size: int = 10):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        if schema and not _SCHEMA_NAME.match(schema):
            raise ValueError(f"invalid schema name: {schema!r}")
        self._schema = schema
        self._local = threading.local()

        def configure(conn):
            if schema:
                conn.execute(f'SET search_path TO "{schema}"')

        self._pool = ConnectionPool(
            url, min_size=min_size, max_size=max_size, open=True, configure=configure,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        with self._pool.connection() as conn:
            if schema:
                conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                conn.execute(f'SET search_path TO "{schema}"')
            # Serialise concurrent first-time schema creation across processes.
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(hashtext('radiodb-schema'))")
                conn.execute(SCHEMA)
        self._check_version()

    def close(self) -> None:
        self._pool.close()

    def _is_unique_violation(self, e: Exception) -> bool:
        from psycopg.errors import UniqueViolation

        return isinstance(e, UniqueViolation)

    def _run(self, sql: str, params: tuple) -> tuple[list[Any], int]:
        pinned = getattr(self._local, "conn", None)
        if pinned is not None:
            return _fetch(pinned.execute(sql, params))
        with self._pool.connection() as conn:
            return _fetch(conn.execute(sql, params))

    @contextmanager
    def transaction(self) -> Iterator[None]:
        pinned = getattr(self._local, "conn", None)
        if pinned is not None:
            with pinned.transaction():  # savepoint
                yield
            return
        with self._pool.connection() as conn:
            self._local.conn = conn
            try:
                with conn.transaction():
                    yield
            finally:
                self._local.conn = None

    def search_text(
        self, user_id: str, query: str, *, speaker_id: str | None = None, limit: int = 20,
        any_terms: bool = False,
    ) -> list[dict]:
        # plainto_tsquery normalises/stems the input and ignores operators, so user text is safe.
        # For any-term matching, its AND (&) is turned into OR (|) and re-parsed with the
        # 'simple' config, which keeps the already-stemmed lexemes as they are.
        tsq = "plainto_tsquery('english', ?)"
        if any_terms:
            tsq = f"to_tsquery('simple', replace({tsq}::text, '&', '|'))"
        sql = (
            "SELECT g.*, s.name AS speaker_name, ts_rank_cd(g.tsv, q) AS rank,"
            f" ts_headline('english', g.text, q, '{_HEADLINE}') AS snippet"
            f" FROM segments g LEFT JOIN speakers s ON s.id = g.speaker_id, {tsq} AS q"
            " WHERE g.tsv @@ q AND g.user_id=?"
        )
        params: list[Any] = [query, user_id]
        if speaker_id:
            sql += " AND g.speaker_id=?"; params.append(speaker_id)
        sql += " ORDER BY rank DESC, g.id LIMIT ?"
        params.append(limit)
        return [_segment(r) for r in self._q(sql, params)]


def _fetch(cur: Any) -> tuple[list[Any], int]:
    return (cur.fetchall() if cur.description else []), cur.rowcount
