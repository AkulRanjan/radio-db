"""Relational catalog (SQLite locally) for recordings, speakers and time-aligned segments.

Audio never lives here — only pointers to blobs, time spans, derived layers and provenance.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
    enrolled    INTEGER NOT NULL DEFAULT 0,   -- 1 = voice profile built from user-provided samples
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
"""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class Catalog:
    def __init__(self, path: Path | str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        self._db.close()

    def _q(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(params)).fetchall()

    def _x(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, tuple(params))

    def transaction(self):
        return _Tx(self)

    # ---- recordings -------------------------------------------------------------------------

    def find_recording_by_hash(self, user_id: str, sha256: str) -> dict | None:
        rows = self._q("SELECT * FROM recordings WHERE user_id=? AND sha256=?", (user_id, sha256))
        return dict(rows[0]) if rows else None

    def add_recording(
        self,
        *,
        user_id: str,
        title: str | None,
        sha256: str,
        blob_key: str,
        format: str,
        sample_rate: int,
        channels: int,
        duration_s: float,
        retention_days: int,
    ) -> dict:
        created = now_utc()
        rec_id = uuid.uuid4().hex
        self._x(
            "INSERT INTO recordings (id, user_id, title, sha256, blob_key, format, sample_rate,"
            " channels, duration_s, created_at, expires_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                rec_id, user_id, title, sha256, blob_key, format, sample_rate, channels,
                duration_s, _iso(created), _iso(created + timedelta(days=retention_days)),
            ),
        )
        return self.get_recording(user_id, rec_id)

    def get_recording(self, user_id: str, rec_id: str) -> dict | None:
        rows = self._q("SELECT * FROM recordings WHERE id=? AND user_id=?", (rec_id, user_id))
        return dict(rows[0]) if rows else None

    def list_recordings(self, user_id: str) -> list[dict]:
        rows = self._q(
            "SELECT * FROM recordings WHERE user_id=? ORDER BY created_at DESC", (user_id,)
        )
        return [dict(r) for r in rows]

    def total_duration(self, user_id: str) -> float:
        return self._q(
            "SELECT COALESCE(SUM(duration_s), 0) AS d FROM recordings WHERE user_id=?", (user_id,)
        )[0]["d"]

    def set_status(self, rec_id: str, status: str, error: str | None = None, producers: dict | None = None):
        self._x(
            "UPDATE recordings SET status=?, error=?, producers=COALESCE(?, producers) WHERE id=?",
            (status, error, json.dumps(producers) if producers else None, rec_id),
        )

    def delete_recording(self, rec_id: str) -> None:
        self._x("DELETE FROM recordings WHERE id=?", (rec_id,))

    def expired_recordings(self, at: datetime | None = None) -> list[dict]:
        rows = self._q("SELECT * FROM recordings WHERE expires_at <= ?", (_iso(at or now_utc()),))
        return [dict(r) for r in rows]

    def blob_in_use(self, blob_key: str) -> bool:
        return bool(self._q("SELECT 1 FROM recordings WHERE blob_key=? LIMIT 1", (blob_key,)))

    # ---- speakers ---------------------------------------------------------------------------

    def add_speaker(
        self, user_id: str, centroid: np.ndarray, *, name: str | None = None,
        enrolled: bool = False, n_samples: int = 1,
    ) -> dict:
        spk_id = uuid.uuid4().hex
        self._x(
            "INSERT INTO speakers (id, user_id, name, enrolled, centroid, n_samples, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (spk_id, user_id, name, int(enrolled), vec_to_blob(centroid), n_samples, _iso(now_utc())),
        )
        return self.get_speaker(user_id, spk_id)

    def get_speaker(self, user_id: str, spk_id: str) -> dict | None:
        rows = self._q("SELECT * FROM speakers WHERE id=? AND user_id=?", (spk_id, user_id))
        return _speaker(rows[0]) if rows else None

    def find_speaker_by_name(self, user_id: str, name: str) -> dict | None:
        rows = self._q(
            "SELECT * FROM speakers WHERE user_id=? AND name=? COLLATE NOCASE", (user_id, name)
        )
        return _speaker(rows[0]) if rows else None

    def list_speakers(self, user_id: str) -> list[dict]:
        rows = self._q(
            "SELECT s.*, (SELECT COUNT(*) FROM segments g WHERE g.speaker_id=s.id) AS n_segments,"
            " (SELECT COALESCE(SUM(g.end_s-g.start_s),0) FROM segments g WHERE g.speaker_id=s.id)"
            " AS talk_time_s FROM speakers s WHERE s.user_id=? ORDER BY s.created_at",
            (user_id,),
        )
        return [_speaker(r) for r in rows]

    def update_speaker(
        self, spk_id: str, *, name: str | None = None, centroid: np.ndarray | None = None,
        n_samples: int | None = None, enrolled: bool | None = None,
    ) -> None:
        sets, params = [], []
        if name is not None:
            sets.append("name=?"); params.append(name)
        if centroid is not None:
            sets.append("centroid=?"); params.append(vec_to_blob(centroid))
        if n_samples is not None:
            sets.append("n_samples=?"); params.append(n_samples)
        if enrolled is not None:
            sets.append("enrolled=?"); params.append(int(enrolled))
        if sets:
            self._x(f"UPDATE speakers SET {', '.join(sets)} WHERE id=?", (*params, spk_id))

    def reassign_segments(self, from_spk: str, to_spk: str) -> None:
        self._x("UPDATE segments SET speaker_id=? WHERE speaker_id=?", (to_spk, from_spk))

    def delete_speaker(self, spk_id: str) -> None:
        # Voice is biometric data: drop the profile and every per-segment voice embedding.
        self._x("UPDATE segments SET embedding=NULL WHERE speaker_id=?", (spk_id,))
        self._x("DELETE FROM speakers WHERE id=?", (spk_id,))

    def delete_orphan_speakers(self, user_id: str) -> int:
        """Remove auto-detected (unnamed, unenrolled) speakers that no longer have any segments."""
        return self._x(
            "DELETE FROM speakers WHERE user_id=? AND name IS NULL AND enrolled=0"
            " AND NOT EXISTS (SELECT 1 FROM segments g WHERE g.speaker_id=speakers.id)",
            (user_id,),
        ).rowcount

    # ---- segments ---------------------------------------------------------------------------

    def replace_segments(self, rec_id: str, user_id: str, segments: list[dict]) -> None:
        with self.transaction():
            self._x("DELETE FROM segments WHERE recording_id=?", (rec_id,))
            for s in segments:
                self._x(
                    "INSERT INTO segments (recording_id, user_id, speaker_id, start_s, end_s, text,"
                    " words, embedding) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        rec_id, user_id, s.get("speaker_id"), s["start_s"], s["end_s"],
                        s.get("text", ""), json.dumps(s.get("words", [])),
                        vec_to_blob(s["embedding"]) if s.get("embedding") is not None else None,
                    ),
                )

    def segments(
        self, user_id: str, *, recording_id: str | None = None, speaker_id: str | None = None
    ) -> list[dict]:
        sql = (
            "SELECT g.*, s.name AS speaker_name FROM segments g"
            " LEFT JOIN speakers s ON s.id=g.speaker_id WHERE g.user_id=?"
        )
        params: list[Any] = [user_id]
        if recording_id:
            sql += " AND g.recording_id=?"; params.append(recording_id)
        if speaker_id:
            sql += " AND g.speaker_id=?"; params.append(speaker_id)
        sql += " ORDER BY g.recording_id, g.start_s"
        return [_segment(r) for r in self._q(sql, params)]

    def search_text(
        self, user_id: str, query: str, *, speaker_id: str | None = None, limit: int = 20
    ) -> list[dict]:
        sql = (
            "SELECT g.*, s.name AS speaker_name, bm25(segments_fts) AS rank,"
            " snippet(segments_fts, 0, '[', ']', '...', 12) AS snippet"
            " FROM segments_fts JOIN segments g ON g.id = segments_fts.rowid"
            " LEFT JOIN speakers s ON s.id = g.speaker_id"
            " WHERE segments_fts MATCH ? AND g.user_id=?"
        )
        params: list[Any] = [_fts_query(query), user_id]
        if speaker_id:
            sql += " AND g.speaker_id=?"; params.append(speaker_id)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        return [_segment(r) for r in self._q(sql, params)]


class _Tx:
    def __init__(self, cat: Catalog):
        self.cat = cat

    def __enter__(self):
        self.cat._lock.acquire()
        self.cat._db.execute("BEGIN")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cat._db.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.cat._lock.release()


def _fts_query(q: str) -> str:
    # Treat user input as plain words (quoted terms, implicit AND) so FTS syntax can't break it.
    terms = [t.replace('"', '""') for t in q.split() if t.strip()]
    return " ".join(f'"{t}"' for t in terms) or '""'


def _speaker(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["centroid"] = blob_to_vec(d["centroid"])
    d["enrolled"] = bool(d["enrolled"])
    return d


def _segment(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["words"] = json.loads(d["words"])
    d["embedding"] = blob_to_vec(d["embedding"]) if d.get("embedding") is not None else None
    return d
