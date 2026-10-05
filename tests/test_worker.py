"""Job queue + worker, on every catalog backend."""

import threading
import time
from collections import Counter
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from radiodb.catalog import _iso, now_utc
from radiodb.worker import SYSTEM_USER, Worker, ensure_purge_scheduled, wait_for
from tests.conftest import VOCAB, conversation, script_words, wav_bytes

SCRIPT = [("A", 3.0), (None, 0.6), ("B", 3.0)]


def _ingest(db, asr, user="u1", seconds=3.0):
    script = [("A", seconds), (None, 0.6), ("B", 3.0)]
    asr.next_words = script_words(script, VOCAB)
    rec, _ = db.ingest(user, wav_bytes(conversation(script)), "c.wav")
    return rec


def _fail_n_times(asr, n, exc=RuntimeError("GPU hiccup")):
    real = asr.transcribe
    calls = {"n": 0}

    def flaky(a):
        calls["n"] += 1
        if calls["n"] <= n:
            raise exc
        return real(a)

    asr.transcribe = flaky
    return calls


def test_queued_processing_end_to_end(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    assert db.get_recording("u1", rec["id"])["status"] == "queued"
    assert db.enqueue_process("u1", rec["id"])["id"] == job["id"]  # idempotent while pending

    assert Worker(db).run_until_idle() == 1
    assert db.get_job("u1", job["id"])["status"] == "done"
    assert db.get_recording("u1", rec["id"])["status"] == "done"
    assert len(db.transcript("u1", rec["id"])) == 2
    # once finished, the same work can be queued again
    assert db.enqueue_process("u1", rec["id"])["id"] != job["id"]


def test_transient_failure_retries_with_backoff(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    _fail_n_times(asr, 1)
    w = Worker(db, backoff_s=60)

    assert w.run_until_idle() == 1  # failed once; retry is not due yet
    j = db.get_job("u1", job["id"])
    assert j["status"] == "queued" and j["attempts"] == 1 and "GPU hiccup" in j["last_error"]
    assert j["run_after"] > _iso(now_utc() + timedelta(seconds=50))
    assert db.get_recording("u1", rec["id"])["status"] == "queued"

    db.catalog._x("UPDATE jobs SET run_after=? WHERE id=?", (_iso(now_utc()), job["id"]))
    assert w.run_until_idle() == 1
    j = db.get_job("u1", job["id"])
    assert j["status"] == "done" and j["attempts"] == 2
    assert db.get_recording("u1", rec["id"])["status"] == "done"


def test_gives_up_after_max_attempts(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    _fail_n_times(asr, 99)
    assert Worker(db, backoff_s=0).run_until_idle() == 3
    j = db.get_job("u1", job["id"])
    assert j["status"] == "failed" and j["attempts"] == 3
    assert db.get_recording("u1", rec["id"])["status"] == "failed"


def test_permanent_errors_are_not_retried(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    _fail_n_times(asr, 99, ValueError("corrupt input"))
    assert Worker(db, backoff_s=0).run_until_idle() == 1
    j = db.get_job("u1", job["id"])
    assert j["status"] == "failed" and j["attempts"] == 1


def test_crashed_worker_job_is_taken_over(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    dead = db.catalog.claim_job("dead-worker", lease_s=0.2)  # claims, then "crashes"
    assert dead["id"] == job["id"]
    assert db.catalog.claim_job("other", lease_s=60) is None  # lease still valid
    time.sleep(0.4)

    assert Worker(db, worker_id="rescuer").run_until_idle() == 1
    j = db.get_job("u1", job["id"])
    assert j["status"] == "done" and j["attempts"] == 2
    assert db.catalog.complete_job(job["id"], "dead-worker") is False  # stale owner can't touch it


def test_heartbeat_keeps_long_job_owned(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    real = asr.transcribe
    asr.transcribe = lambda a: (time.sleep(1.5), real(a))[1]

    w = Worker(db, worker_id="slow", lease_s=0.5, heartbeat_s=0.1)
    t = threading.Thread(target=w.run_until_idle)
    t.start()
    time.sleep(0.9)  # well past the original lease
    assert db.catalog.claim_job("thief", lease_s=60) is None
    t.join()
    j = db.get_job("u1", job["id"])
    assert j["status"] == "done" and j["attempts"] == 1


def test_concurrent_workers_run_each_job_once(db, asr):
    recs = [_ingest(db, asr, user=f"u{i % 2}", seconds=2.0 + i * 0.1) for i in range(8)]
    for r in recs:
        db.enqueue_process(r["user_id"], r["id"])
    calls = Counter()
    real_process = db.process

    def counting(user, rec_id):
        calls[rec_id] += 1
        return real_process(user, rec_id)

    db.process = counting
    workers = [Worker(db, worker_id=f"w{i}", poll_s=0.05) for i in range(4)]
    threads = [threading.Thread(target=w.run_until_idle) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls == Counter({r["id"]: 1 for r in recs})
    assert all(db.get_recording(r["user_id"], r["id"])["status"] == "done" for r in recs)


def test_purge_job_runs_and_reschedules_daily(db, asr):
    old = _ingest(db, asr)
    db.catalog._x("UPDATE recordings SET expires_at=? WHERE id=?", (_iso(now_utc() - timedelta(days=1)), old["id"]))
    ensure_purge_scheduled(db)
    ensure_purge_scheduled(db)  # idempotent
    assert len(db.catalog.list_jobs(SYSTEM_USER, "queued")) == 1

    assert Worker(db).run_until_idle() == 1
    assert db.catalog.get_recording("u1", old["id"]) is None
    pending = db.catalog.list_jobs(SYSTEM_USER, "queued")
    assert len(pending) == 1 and pending[0]["run_after"] > _iso(now_utc() + timedelta(hours=23))


def test_deleting_recording_drops_its_jobs(db, asr):
    rec = _ingest(db, asr)
    job = db.enqueue_process("u1", rec["id"])
    db.delete_recording("u1", rec["id"])
    assert db.catalog.get_job("u1", job["id"]) is None


def test_enrich_job_validates_up_front(db, asr):
    rec = _ingest(db, asr)
    with pytest.raises(ValueError, match="process it first"):
        db.enqueue_enrich("u1", rec["id"])
    db.process("u1", rec["id"])
    with pytest.raises(ValueError, match="unknown"):
        db.enqueue_enrich("u1", rec["id"], ["telepathy"])
    job = db.enqueue_enrich("u1", rec["id"], ["prosody"])
    assert Worker(db).run_until_idle() == 1 and db.get_job("u1", job["id"])["status"] == "done"


def test_migrates_v1_catalog(catalog):
    catalog._x("UPDATE radiodb_meta SET value='1' WHERE key='schema_version'")
    catalog._check_version()
    assert catalog._q("SELECT value FROM radiodb_meta WHERE key='schema_version'")[0]["value"] == "2"


def test_api_inline_worker_processes_uploads(db, asr):
    from radiodb.api import create_app

    asr.next_words = script_words(SCRIPT, VOCAB)
    with TestClient(create_app(db, inline_worker=True)) as client:  # lifespan starts the worker
        r = client.post("/recordings", headers={"X-User-Id": "u1"},
                        files={"file": ("c.wav", wav_bytes(conversation(SCRIPT)), "audio/wav")})
        job = wait_for(db, "u1", r.json()["job"]["id"], timeout=30, poll_s=0.1)
        assert job["status"] == "done"
        assert client.get("/jobs", headers={"X-User-Id": "u1"}).json()[0]["id"] == job["id"]
