"""Background worker: takes jobs from the catalog's `jobs` table and runs them.

Run as many as you like (``radiodb worker``), on any machines that can reach the catalog and
blob store. Postgres hands each job to exactly one worker (``FOR UPDATE SKIP LOCKED``); SQLite
does it with a single atomic UPDATE.

- **Leases.** A claimed job is leased for ``lease_s``. A heartbeat thread keeps extending it while
  the job runs, so long recordings aren't stolen. If a worker dies, the lease expires and
  another worker picks the job up.
- **Retries.** Transient failures are retried with exponential backoff up to the job's
  ``max_attempts``. Permanent ones (bad audio, missing recording, invalid request) fail at once.
- **Retention.** A singleton ``purge`` job runs ``purge_expired`` and reschedules itself daily.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from datetime import timedelta

from .audio import UnsupportedAudio
from .catalog import now_utc
from .service import NotFound, RadioDB

log = logging.getLogger(__name__)

PERMANENT_ERRORS = (NotFound, UnsupportedAudio, ValueError)
PURGE_EVERY = timedelta(days=1)
SYSTEM_USER = "_system"


class Worker:
    def __init__(
        self, db: RadioDB, worker_id: str | None = None, *, lease_s: float = 600, heartbeat_s: float = 60,
        poll_s: float = 2.0, backoff_s: float = 30, kinds: list[str] | None = None,
    ):
        self.db = db
        self.id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self.lease_s, self.heartbeat_s, self.poll_s, self.backoff_s = lease_s, heartbeat_s, poll_s, backoff_s
        self.kinds = kinds
        self.stop_event = threading.Event()

    # ---- loop ---------------------------------------------------------------------------------

    def run(self, max_jobs: int | None = None) -> int:
        """Process jobs until stopped (or ``max_jobs`` done). Returns the number of jobs run."""
        log.info("worker %s started", self.id)
        ensure_purge_scheduled(self.db)
        done = 0
        while not self.stop_event.is_set() and (max_jobs is None or done < max_jobs):
            if self.run_once():
                done += 1
            else:
                self.stop_event.wait(self.poll_s)
        log.info("worker %s stopped after %d job(s)", self.id, done)
        return done

    def run_until_idle(self) -> int:
        """Run jobs that are due right now, then return (tests, scripts, cron)."""
        done = 0
        while self.run_once():
            done += 1
        return done

    def stop(self) -> None:
        self.stop_event.set()

    def run_once(self) -> bool:
        cat = self.db.catalog
        cat.fail_abandoned_jobs()
        job = cat.claim_job(self.id, self.lease_s, self.kinds)
        if job is None:
            return False
        log.info("job %s (%s, attempt %d/%d) claimed", job["id"], job["kind"], job["attempts"], job["max_attempts"])

        stop_hb = threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(job["id"], stop_hb), daemon=True)
        hb.start()
        try:
            self._execute(job)
        except Exception as e:  # noqa: BLE001 — every failure is recorded on the job
            permanent = isinstance(e, PERMANENT_ERRORS) or job["attempts"] >= job["max_attempts"]
            retry_at = None if permanent else now_utc() + timedelta(
                seconds=self.backoff_s * 2 ** (job["attempts"] - 1))
            log.warning("job %s failed (%s): %s", job["id"], "permanent" if permanent else f"retry at {retry_at}", e)
            cat.fail_job(job["id"], self.id, f"{type(e).__name__}: {e}", retry_at)
            if retry_at and job["recording_id"]:
                cat.set_status(job["recording_id"], "queued", error=f"{type(e).__name__}: {e} (will retry)")
        else:
            cat.complete_job(job["id"], self.id)
            log.info("job %s done", job["id"])
            if job["kind"] == "purge":  # reschedule only once this run is no longer active
                cat.enqueue_job(user_id=SYSTEM_USER, kind="purge", run_after=now_utc() + PURGE_EVERY)
        finally:
            stop_hb.set()
            hb.join()
        return True

    def _heartbeat(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_s):
            if not self.db.catalog.heartbeat_job(job_id, self.id, self.lease_s):
                log.warning("lost lease on job %s", job_id)
                return

    # ---- job kinds ------------------------------------------------------------------------------

    def _execute(self, job: dict) -> None:
        kind, user, rec = job["kind"], job["user_id"], job["recording_id"]
        if kind == "process":
            self.db.process(user, rec)
        elif kind == "enrich":
            self.db.enrich(user, rec, job["payload"].get("layers"))
        elif kind == "purge":
            log.info("purged %d expired recording(s)", self.db.purge_expired())
        else:
            raise ValueError(f"unknown job kind: {kind}")


def ensure_purge_scheduled(db: RadioDB) -> None:
    """Make sure exactly one retention purge is pending (idempotent across workers)."""
    db.catalog.enqueue_job(user_id=SYSTEM_USER, kind="purge")


class BackgroundWorker:
    """Runs a Worker in a daemon thread (the API's inline worker for local development)."""

    def __init__(self, db: RadioDB, **kwargs):
        self.worker = Worker(db, **kwargs)
        self.thread = threading.Thread(target=self.worker.run, name="radiodb-worker", daemon=True)

    def start(self) -> "BackgroundWorker":
        self.thread.start()
        return self

    def stop(self, timeout: float = 30) -> None:
        self.worker.stop()
        self.thread.join(timeout)


def wait_for(db: RadioDB, user_id: str, job_id: str, timeout: float = 600, poll_s: float = 0.5) -> dict:
    """Block until a job finishes (done/failed); returns the job."""
    deadline = time.monotonic() + timeout
    while True:
        job = db.catalog.get_job(user_id, job_id)
        if job is None or job["status"] in ("done", "failed") or time.monotonic() > deadline:
            return job
        time.sleep(poll_s)
