"""SQLite catalog: zero-setup local development (WAL mode, FTS5 full-text search)."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .base import BaseCatalog, _segment

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS recordings (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    title        TEXT,
    sha256       TEXT NOT NULL,
    blob_key     TEXT NOT NULL,
    format       TEXT NOT NULL,
    sample_rate  INTEGER NOT NULL,
    channels     INTEGER NOT NULL,
    duration_s   REAL NOT NULL,
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | processing | done | failed
    error        TEXT,
    producers    TEXT,                              -- JSON: model names/versions that built the layers
    UNIQUE (user_id, sha256)
);
CREATE INDEX IF NOT EXISTS ix_recordings_user ON recordings(user_id);
CREATE INDEX IF NOT EXISTS ix_recordings_expires ON recordings(expires_at);

CREATE TABLE IF NOT EXISTS speakers (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    name        TEXT,
    enrolled    BOOLEAN NOT NULL DEFAULT 0,   -- voice profile built from user-provided samples
    centroid    BLOB NOT NULL,                -- float32 L2-normalised speaker embedding
    n_samples   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_speakers_user ON speakers(user_id);

CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id  TEXT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    speaker_id    TEXT REFERENCES speakers(id) ON DELETE SET NULL,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    text          TEXT NOT NULL DEFAULT '',
    words         TEXT NOT NULL DEFAULT '[]',   -- JSON [{w, start, end, p}]
    embedding     BLOB                          -- speaker embedding of this segment
);
CREATE INDEX IF NOT EXISTS ix_segments_recording ON segments(recording_id, start_s);
CREATE INDEX IF NOT EXISTS ix_segments_speaker ON segments(speaker_id);
CREATE INDEX IF NOT EXISTS ix_segments_user ON segments(user_id);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text, content='segments', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE OF text ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;

-- Derived per-segment layers (prosody, semantic text embedding, ...). One row per layer so
-- new layers can be backfilled without touching audio, ASR or diarization.
CREATE TABLE IF NOT EXISTS segment_layers (
    segment_id  INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    layer       TEXT NOT NULL,
    producer    TEXT NOT NULL,   -- model/tool name and version that built this row
    data        TEXT,            -- JSON features
    vector      BLOB,            -- optional float32 embedding
    PRIMARY KEY (segment_id, layer)
);
CREATE INDEX IF NOT EXISTS ix_segment_layers_layer ON segment_layers(layer);
-- Work queue (schema v2). One active (queued/running) job per recording and kind.
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    kind          TEXT NOT NULL,                -- process | enrich | purge
    recording_id  TEXT REFERENCES recordings(id) ON DELETE CASCADE,   -- NULL for system jobs
    payload       TEXT NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL,                -- queued | running | done | failed
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 3,
    run_after     TEXT NOT NULL,
    locked_by     TEXT,
    locked_until  TEXT,
    last_error    TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    finished_at   TEXT
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


class SQLiteCatalog(BaseCatalog):
    def __init__(self, path: Path | str):
        self._lock = threading.RLock()  # one connection shared by API worker threads
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._depth = 0
        with self._lock:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(SCHEMA)
        self._check_version()

    def close(self) -> None:
        self._db.close()

    def _is_unique_violation(self, e: Exception) -> bool:
        return isinstance(e, sqlite3.IntegrityError) and "UNIQUE" in str(e)

    def _run(self, sql: str, params: tuple) -> tuple[list[Any], int]:
        with self._lock:
            cur = self._db.execute(sql, params)
            return (cur.fetchall() if cur.description else []), cur.rowcount

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            outer = self._depth == 0
            if outer:
                self._db.execute("BEGIN")
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outer:
                    self._db.execute("ROLLBACK")
                raise
            self._depth -= 1
            if outer:
                self._db.execute("COMMIT")

    def search_text(
        self, user_id: str, query: str, *, speaker_id: str | None = None, limit: int = 20,
        any_terms: bool = False,
    ) -> list[dict]:
        sql = (
            "SELECT g.*, s.name AS speaker_name, bm25(segments_fts) AS rank,"
            " snippet(segments_fts, 0, '[', ']', '...', 12) AS snippet"
            " FROM segments_fts JOIN segments g ON g.id = segments_fts.rowid"
            " LEFT JOIN speakers s ON s.id = g.speaker_id"
            " WHERE segments_fts MATCH ? AND g.user_id=?"
        )
        params: list[Any] = [_fts_query(query, any_terms), user_id]
        if speaker_id:
            sql += " AND g.speaker_id=?"; params.append(speaker_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        return [_segment(r) for r in self._q(sql, params)]


def _fts_query(q: str, any_terms: bool = False) -> str:
    # Treat user input as plain quoted words so FTS syntax can't break it. All words must match
    # (AND) by default; any_terms=True ORs them and lets BM25 rank segments matching more words.
    terms = [t.replace('"', '""') for t in q.split() if t.strip()]
    return (" OR " if any_terms else " ").join(f'"{t}"' for t in terms) or '""'
