"""Backend-neutral catalog logic for recordings, speakers, segments and derived layers.

Audio never lives here — only pointers to blobs, time spans, derived layers and provenance.
SQL is written once with ``?`` placeholders; each dialect supplies the connection, schema,
placeholder style and full-text search.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator

import numpy as np


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes | memoryview) -> np.ndarray:
    return np.frombuffer(bytes(b), dtype=np.float32)


def _json(v: Any) -> Any:
    """JSON columns come back as text (SQLite) or already parsed (Postgres jsonb)."""
    return json.loads(v) if isinstance(v, str) else v


SCHEMA_VERSION = 2  # bump together with a step in BaseCatalog._migrate

# v1 -> v2: `jobs` table (work queue). It is created by each dialect's CREATE ... IF NOT EXISTS
# schema, so the migration step itself has nothing left to do.

ACTIVE_JOB = "('queued','running')"


class SchemaVersionError(RuntimeError):
    pass


class BaseCatalog:
    placeholder = "?"
    skip_locked = ""  # Postgres: " FOR UPDATE SKIP LOCKED" so concurrent workers never collide

    # ---- dialect hooks ----------------------------------------------------------------------

    def _run(self, sql: str, params: tuple) -> tuple[list[Any], int]:
        """Execute one statement; return (fetched rows or [], rowcount). Thread-safe."""
        raise NotImplementedError

    @contextmanager
    def transaction(self) -> Iterator[None]:
        raise NotImplementedError
        yield

    def close(self) -> None:
        raise NotImplementedError

    def search_text(
        self, user_id: str, query: str, *, speaker_id: str | None = None, limit: int = 20,
        any_terms: bool = False,
    ) -> list[dict]:
        """Full-text search. All words must match unless ``any_terms`` (then rank by matches).
        Rows carry ``snippet`` with matches wrapped in [brackets]."""
        raise NotImplementedError

    # ---- plumbing -----------------------------------------------------------------------------

    def _sql(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def _q(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        rows, _ = self._run(self._sql(sql), tuple(params))
        return [self._row(r) for r in rows]

    def _x(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Execute a statement; return the affected row count."""
        return self._run(self._sql(sql), tuple(params))[1]

    def _check_version(self) -> None:
        """Record the schema version on first use, migrate older schemas forward step by step,
        and refuse to run against a schema newer than this code."""
        with self.transaction():
            rows = self._q("SELECT value FROM radiodb_meta WHERE key='schema_version'")
            if not rows:
                self._x("INSERT INTO radiodb_meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
                return
            current = int(rows[0]["value"])
            if current > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"catalog schema v{current} is newer than this radiodb (v{SCHEMA_VERSION}); upgrade radiodb"
                )
            for v in range(current + 1, SCHEMA_VERSION + 1):
                self._migrate(v)
            if current < SCHEMA_VERSION:
                self._x("UPDATE radiodb_meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION),))

    def _is_unique_violation(self, e: Exception) -> bool:
        raise NotImplementedError

    def _migrate(self, to_version: int) -> None:
        """Bring the schema from ``to_version - 1`` to ``to_version``."""
        if to_version == 2:
            return  # jobs table: already created by the dialect schema
        raise SchemaVersionError(f"no migration to v{to_version}")

    def _row(self, r: Any) -> dict:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, datetime):
                d[k] = _iso(v)
        return d

    # ---- recordings -------------------------------------------------------------------------

    def find_recording_by_hash(self, user_id: str, sha256: str) -> dict | None:
        rows = self._q("SELECT * FROM recordings WHERE user_id=? AND sha256=?", (user_id, sha256))
        return _recording(rows[0]) if rows else None

    def add_recording(
        self, *, user_id: str, title: str | None, sha256: str, blob_key: str, format: str,
        sample_rate: int, channels: int, duration_s: float, retention_days: int,
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
        return _recording(rows[0]) if rows else None

    def list_recordings(self, user_id: str) -> list[dict]:
        rows = self._q("SELECT * FROM recordings WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        return [_recording(r) for r in rows]

    def total_duration(self, user_id: str) -> float:
        return float(self._q(
            "SELECT COALESCE(SUM(duration_s), 0) AS d FROM recordings WHERE user_id=?", (user_id,)
        )[0]["d"])

    def set_status(self, rec_id: str, status: str, error: str | None = None, producers: dict | None = None):
        self._x(
            "UPDATE recordings SET status=?, error=?, producers=COALESCE(?, producers) WHERE id=?",
            (status, error, json.dumps(producers) if producers else None, rec_id),
        )

    def delete_recording(self, rec_id: str) -> None:
        self._x("DELETE FROM recordings WHERE id=?", (rec_id,))

    def expired_recordings(self, at: datetime | None = None) -> list[dict]:
        rows = self._q("SELECT * FROM recordings WHERE expires_at <= ?", (_iso(at or now_utc()),))
        return [_recording(r) for r in rows]

    def blob_in_use(self, blob_key: str) -> bool:
        return bool(self._q("SELECT 1 AS one FROM recordings WHERE blob_key=? LIMIT 1", (blob_key,)))

    # ---- speakers ---------------------------------------------------------------------------

    def add_speaker(
        self, user_id: str, centroid: np.ndarray, *, name: str | None = None,
        enrolled: bool = False, n_samples: int = 1,
    ) -> dict:
        spk_id = uuid.uuid4().hex
        self._x(
            "INSERT INTO speakers (id, user_id, name, enrolled, centroid, n_samples, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (spk_id, user_id, name, bool(enrolled), vec_to_blob(centroid), n_samples, _iso(now_utc())),
        )
        return self.get_speaker(user_id, spk_id)

    def get_speaker(self, user_id: str, spk_id: str) -> dict | None:
        rows = self._q("SELECT * FROM speakers WHERE id=? AND user_id=?", (spk_id, user_id))
        return _speaker(rows[0]) if rows else None

    def find_speaker_by_name(self, user_id: str, name: str) -> dict | None:
        rows = self._q("SELECT * FROM speakers WHERE user_id=? AND lower(name)=lower(?)", (user_id, name))
        return _speaker(rows[0]) if rows else None

    def list_speakers(self, user_id: str) -> list[dict]:
        rows = self._q(
            "SELECT s.*, (SELECT COUNT(*) FROM segments g WHERE g.speaker_id=s.id) AS n_segments,"
            " (SELECT COALESCE(SUM(g.end_s-g.start_s),0) FROM segments g WHERE g.speaker_id=s.id)"
            " AS talk_time_s FROM speakers s WHERE s.user_id=? ORDER BY s.created_at, s.id",
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
            sets.append("enrolled=?"); params.append(bool(enrolled))
        if sets:
            self._x(f"UPDATE speakers SET {', '.join(sets)} WHERE id=?", (*params, spk_id))

    def reassign_segments(self, from_spk: str, to_spk: str) -> None:
        self._x("UPDATE segments SET speaker_id=? WHERE speaker_id=?", (to_spk, from_spk))

    def delete_speaker(self, spk_id: str) -> None:
        # Voice is biometric data: drop the profile, every per-segment voice embedding and the
        # voice-derived layers (prosody, emotion) of that speaker's segments.
        with self.transaction():
            self._x("UPDATE segments SET embedding=NULL WHERE speaker_id=?", (spk_id,))
            self._x(
                "DELETE FROM segment_layers WHERE layer IN ('prosody', 'emotion')"
                " AND segment_id IN (SELECT id FROM segments WHERE speaker_id=?)",
                (spk_id,),
            )
            self._x("DELETE FROM speakers WHERE id=?", (spk_id,))

    def delete_orphan_speakers(self, user_id: str) -> int:
        """Remove auto-detected (unnamed, unenrolled) speakers that no longer have any segments."""
        return self._x(
            "DELETE FROM speakers WHERE user_id=? AND name IS NULL AND NOT enrolled"
            " AND NOT EXISTS (SELECT 1 FROM segments g WHERE g.speaker_id=speakers.id)",
            (user_id,),
        )

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
        sql += " ORDER BY g.recording_id, g.start_s, g.id"
        return [_segment(r) for r in self._q(sql, params)]

    def put_layer(self, layer: str, producer: str, rows: list[tuple[int, dict | None, np.ndarray | None]]) -> None:
        """Upsert one layer for many segments: rows of (segment_id, data, vector)."""
        with self.transaction():
            for seg_id, data, vec in rows:
                self._x(
                    "INSERT INTO segment_layers (segment_id, layer, producer, data, vector)"
                    " VALUES (?,?,?,?,?) ON CONFLICT (segment_id, layer) DO UPDATE SET"
                    " producer=excluded.producer, data=excluded.data, vector=excluded.vector",
                    (
                        seg_id, layer, producer, json.dumps(data) if data is not None else None,
                        vec_to_blob(vec) if vec is not None else None,
                    ),
                )

    def layers(
        self, user_id: str, layer: str, *, recording_id: str | None = None, speaker_id: str | None = None
    ) -> dict[int, dict]:
        """{segment_id: {"producer", "data", "vector"}} for one layer, scoped to a user."""
        sql = (
            "SELECT l.segment_id, l.producer, l.data, l.vector FROM segment_layers l"
            " JOIN segments g ON g.id = l.segment_id WHERE g.user_id=? AND l.layer=?"
        )
        params: list[Any] = [user_id, layer]
        if recording_id:
            sql += " AND g.recording_id=?"; params.append(recording_id)
        if speaker_id:
            sql += " AND g.speaker_id=?"; params.append(speaker_id)
        return {
            r["segment_id"]: {
                "producer": r["producer"],
                "data": _json(r["data"]),
                "vector": blob_to_vec(r["vector"]) if r["vector"] is not None else None,
            }
            for r in self._q(sql, params)
        }


    # ---- jobs (work queue) -------------------------------------------------------------------

    def enqueue_job(
        self, *, user_id: str, kind: str, recording_id: str | None = None, payload: dict | None = None,
        max_attempts: int = 3, run_after: datetime | None = None,
    ) -> tuple[dict, bool]:
        """Queue a job. At most one queued/running job per (recording, kind): if one exists it is
        returned instead. Returns (job, created)."""
        existing = self._active_job(recording_id, kind)
        if existing:
            return existing, False
        now = now_utc()
        job_id = uuid.uuid4().hex
        try:
            self._x(
                "INSERT INTO jobs (id, user_id, kind, recording_id, payload, status, attempts, max_attempts,"
                " run_after, created_at, updated_at) VALUES (?,?,?,?,?,'queued',0,?,?,?,?)",
                (job_id, user_id, kind, recording_id, json.dumps(payload or {}), max_attempts,
                 _iso(run_after or now), _iso(now), _iso(now)),
            )
        except Exception as e:  # lost a race with a concurrent enqueue of the same job
            if not self._is_unique_violation(e):
                raise
            return self._active_job(recording_id, kind), False
        return self._job_by_id(job_id), True

    def _active_job(self, recording_id: str | None, kind: str) -> dict | None:
        rows = self._q(
            f"SELECT * FROM jobs WHERE COALESCE(recording_id,'')=? AND kind=? AND status IN {ACTIVE_JOB}",
            (recording_id or "", kind),
        )
        return _job(rows[0]) if rows else None

    def _job_by_id(self, job_id: str) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE id=?", (job_id,))
        return _job(rows[0]) if rows else None

    def get_job(self, user_id: str, job_id: str) -> dict | None:
        rows = self._q("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id))
        return _job(rows[0]) if rows else None

    def list_jobs(self, user_id: str | None = None, status: str | None = None, limit: int = 100) -> list[dict]:
        sql, params = "SELECT * FROM jobs WHERE 1=1", []
        if user_id:
            sql += " AND user_id=?"; params.append(user_id)
        if status:
            sql += " AND status=?"; params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return [_job(r) for r in self._q(sql, params)]

    def job_counts(self, user_id: str | None = None) -> dict[str, int]:
        sql, params = "SELECT status, COUNT(*) AS n FROM jobs", []
        if user_id:
            sql += " WHERE user_id=?"; params.append(user_id)
        return {r["status"]: int(r["n"]) for r in self._q(sql + " GROUP BY status", params)}

    def claim_job(self, worker_id: str, lease_s: float, kinds: list[str] | None = None) -> dict | None:
        """Atomically take the next runnable job: queued and due, or running with an expired lease
        (its worker died). Returns it with ``attempts`` already incremented."""
        now = now_utc()
        kind_sql = ""
        params: list[Any] = [worker_id, _iso(now + timedelta(seconds=lease_s)), _iso(now), _iso(now), _iso(now)]
        if kinds:
            kind_sql = f" AND kind IN ({','.join('?' * len(kinds))})"
            params += kinds
        rows = self._q(
            "UPDATE jobs SET status='running', attempts=attempts+1, locked_by=?, locked_until=?, updated_at=?"
            " WHERE id = (SELECT id FROM jobs"
            "   WHERE ((status='queued' AND run_after<=?) OR (status='running' AND locked_until<?))"
            f"   AND attempts < max_attempts{kind_sql}"
            f"   ORDER BY run_after, created_at LIMIT 1{self.skip_locked})"
            " RETURNING *",
            params,
        )
        return _job(rows[0]) if rows else None

    def fail_abandoned_jobs(self) -> int:
        """Running jobs whose worker vanished and that have no attempts left become failed."""
        now = _iso(now_utc())
        return self._x(
            "UPDATE jobs SET status='failed', finished_at=?, updated_at=?, locked_by=NULL,"
            " last_error=COALESCE(last_error, 'worker lost') || ' (lease expired; attempts exhausted)'"
            " WHERE status='running' AND locked_until<? AND attempts>=max_attempts",
            (now, now, now),
        )

    def heartbeat_job(self, job_id: str, worker_id: str, lease_s: float) -> bool:
        """Extend a running job's lease. False means this worker no longer owns the job."""
        now = now_utc()
        return self._x(
            "UPDATE jobs SET locked_until=?, updated_at=? WHERE id=? AND locked_by=? AND status='running'",
            (_iso(now + timedelta(seconds=lease_s)), _iso(now), job_id, worker_id),
        ) == 1

    def complete_job(self, job_id: str, worker_id: str) -> bool:
        now = _iso(now_utc())
        return self._x(
            "UPDATE jobs SET status='done', finished_at=?, updated_at=?, locked_by=NULL, locked_until=NULL"
            " WHERE id=? AND locked_by=? AND status='running'",
            (now, now, job_id, worker_id),
        ) == 1

    def fail_job(self, job_id: str, worker_id: str, error: str, retry_at: datetime | None) -> bool:
        """Record a failure: requeue for ``retry_at``, or mark failed for good when None."""
        now = _iso(now_utc())
        if retry_at is not None:
            return self._x(
                "UPDATE jobs SET status='queued', run_after=?, last_error=?, updated_at=?, locked_by=NULL,"
                " locked_until=NULL WHERE id=? AND locked_by=? AND status='running'",
                (_iso(retry_at), error, now, job_id, worker_id),
            ) == 1
        return self._x(
            "UPDATE jobs SET status='failed', last_error=?, finished_at=?, updated_at=?, locked_by=NULL,"
            " locked_until=NULL WHERE id=? AND locked_by=? AND status='running'",
            (error, now, now, job_id, worker_id),
        ) == 1


def _job(d: dict) -> dict:
    d["payload"] = _json(d.get("payload")) or {}
    return d


def _recording(d: dict) -> dict:
    # producers is exposed as a JSON string on every backend
    if d.get("producers") is not None and not isinstance(d["producers"], str):
        d["producers"] = json.dumps(d["producers"])
    return d


def _speaker(d: dict) -> dict:
    d["centroid"] = blob_to_vec(d["centroid"])
    d["enrolled"] = bool(d["enrolled"])
    return d


def _segment(d: dict) -> dict:
    d.pop("tsv", None)  # Postgres full-text column
    d["words"] = _json(d["words"])
    d["embedding"] = blob_to_vec(d["embedding"]) if d.get("embedding") is not None else None
    return d
