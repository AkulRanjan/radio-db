"""HTTP API. Authentication: see ``radiodb.auth`` (``RADIODB_AUTH=jwt`` in production)."""


from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .audio import UnsupportedAudio
from .auth import AuthError, Authenticator
from .service import NotFound, QuotaExceeded, RadioDB

UPLOAD_CHUNK = 1 << 20

MEDIA_TYPES = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}


class RenameBody(BaseModel):
    name: str


class MergeBody(BaseModel):
    keep: str
    drop: str


def create_app(db: RadioDB | None = None, inline_worker: bool | None = None) -> FastAPI:
    """``inline_worker`` runs a job worker thread inside the API process (local development;
    defaults to RADIODB_INLINE_WORKER). In production run ``radiodb worker`` processes instead."""
    state = {"db": db}

    def get_db() -> RadioDB:
        if state["db"] is None:
            state["db"] = RadioDB()
        return state["db"]

    @asynccontextmanager
    async def lifespan(_app):
        from .worker import BackgroundWorker

        run_inline = get_db().settings.inline_worker if inline_worker is None else inline_worker
        bg = BackgroundWorker(get_db(), poll_s=1.0).start() if run_inline else None
        yield
        if bg:
            bg.stop()

    app = FastAPI(title="radio-db", version="0.1.0", lifespan=lifespan)

    DB = Annotated[RadioDB, Depends(get_db)]

    def get_auth() -> Authenticator:
        if "auth" not in state:
            state["auth"] = Authenticator(get_db().settings)
        return state["auth"]

    def get_user(request: Request) -> str:
        try:
            return get_auth().user_id(request.headers.get("authorization"), request.headers.get("x-user-id"))
        except AuthError as e:
            raise HTTPException(401, str(e), headers={"WWW-Authenticate": "Bearer"})

    User = Annotated[str, Depends(get_user)]

    async def read_limited(file: UploadFile, db: RadioDB) -> bytes:
        limit = db.settings.max_upload_mb * 1024 * 1024
        chunks, size = [], 0
        while chunk := await file.read(UPLOAD_CHUNK):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"upload exceeds {db.settings.max_upload_mb} MB")
            chunks.append(chunk)
        return b"".join(chunks)

    @app.get("/health")
    def health(db: DB):
        """Unauthenticated readiness check: catalog and blob store reachable."""
        h = db.health()
        return JSONResponse(h, status_code=200 if h["status"] == "ok" else 503)

    @app.exception_handler(NotFound)
    async def _nf(_req, exc):
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})

    @app.post("/recordings", status_code=201)
    async def upload(
        user: User, db: DB,
        file: UploadFile = File(...), title: str | None = Form(None), process: bool = Form(True),
    ):
        """Store audio. Processing is queued; poll ``GET /jobs/{job.id}`` or the recording's status."""
        data = await read_limited(file, db)
        try:
            rec, created = db.ingest(user, data, title or file.filename)
        except UnsupportedAudio as e:
            raise HTTPException(415, str(e))
        except QuotaExceeded as e:
            raise HTTPException(403, str(e))
        job = None
        if process and (created or rec["status"] in ("pending", "failed")):
            job = db.enqueue_process(user, rec["id"])
            rec = db.get_recording(user, rec["id"])
        return {"recording": rec, "created": created, "job": job}

    @app.get("/recordings")
    def list_recordings(user: User, db: DB):
        return db.list_recordings(user)

    @app.get("/recordings/{rec_id}")
    def get_recording(user: User, db: DB, rec_id: str):
        return db.get_recording(user, rec_id)

    @app.post("/recordings/{rec_id}/process", status_code=202)
    def reprocess(user: User, db: DB, rec_id: str):
        """Queue a full rebuild (diarization, identification, ASR, layers)."""
        return db.enqueue_process(user, rec_id)

    @app.post("/recordings/{rec_id}/enrich", status_code=202)
    def enrich(user: User, db: DB, rec_id: str, layer: list[str] | None = Query(None)):
        """Queue a rebuild of derived layers (all enabled, or the named ones) without re-running ASR."""
        try:
            return db.enqueue_enrich(user, rec_id, layer)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/jobs")
    def list_jobs(
        user: User, db: DB, status: Literal["queued", "running", "done", "failed"] | None = None,
        limit: int = Query(50, ge=1, le=500),
    ):
        return db.list_jobs(user, status, limit)

    @app.get("/jobs/{job_id}")
    def get_job(user: User, db: DB, job_id: str):
        return db.get_job(user, job_id)

    @app.delete("/recordings/{rec_id}", status_code=204)
    def delete_recording(user: User, db: DB, rec_id: str):
        db.delete_recording(user, rec_id)

    @app.get("/recordings/{rec_id}/transcript")
    def transcript(user: User, db: DB, rec_id: str):
        return db.transcript(user, rec_id)

    @app.get("/recordings/{rec_id}/audio")
    def get_audio(
        user: User, db: DB, rec_id: str,
        start: float | None = Query(None, ge=0), end: float | None = Query(None, gt=0),
    ):
        if start is None and end is None:
            data, fmt = db.original(user, rec_id)
            return Response(data, media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"))
        rec = db.get_recording(user, rec_id)
        try:
            wav = db.clip(user, rec_id, start or 0.0, end if end is not None else rec["duration_s"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        return Response(wav, media_type="audio/wav")

    @app.get("/speakers")
    def speakers(user: User, db: DB):
        return db.list_speakers(user)

    @app.post("/speakers/enroll", status_code=201)
    async def enroll(user: User, db: DB, name: str = Form(...), files: list[UploadFile] = File(...)):
        try:
            return db.enroll(user, name, [await read_limited(f, db) for f in files])
        except (UnsupportedAudio, ValueError) as e:
            raise HTTPException(422, str(e))

    @app.patch("/speakers/{spk_id}")
    def rename(user: User, db: DB, spk_id: str, body: RenameBody):
        try:
            return db.rename_speaker(user, spk_id, body.name)
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/speakers/merge")
    def merge(user: User, db: DB, body: MergeBody):
        try:
            return db.merge_speakers(user, body.keep, body.drop)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/speakers/{spk_id}", status_code=204)
    def delete_speaker(user: User, db: DB, spk_id: str):
        db.delete_speaker(user, spk_id)

    @app.get("/speakers/{ref}/profile")
    def speaker_profile(user: User, db: DB, ref: str):
        return db.speaker_profile(user, ref)

    @app.get("/speakers/{ref}/segments")
    def speaker_segments(user: User, db: DB, ref: str):
        return db.speaker_segments(user, ref)

    @app.get("/search")
    def search(
        user: User, db: DB,
        q: str | None = Query(None, description="words or a description; omit to filter only"),
        mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
        speaker: str | None = Query(None, description="speaker id or name"),
        recording: str | None = None,
        pitch: Literal["low", "typical", "high"] | None = None,
        pace: Literal["slow", "typical", "fast"] | None = None,
        volume: Literal["quiet", "typical", "loud"] | None = None,
        expressiveness: Literal["flat", "typical", "animated"] | None = None,
        emotion: Literal["neutral", "happy", "angry", "sad"] | None = None,
        limit: int = Query(20, ge=1, le=500),
    ):
        """Search segments. Style filters are relative to each speaker's own baseline."""
        style = {k: v for k, v in
                 {"pitch": pitch, "pace": pace, "volume": volume, "expressiveness": expressiveness}.items() if v}
        try:
            return db.search(user, q, mode=mode, speaker=speaker, recording=recording, style=style,
                             emotion=emotion, limit=limit)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/search/voice")
    async def search_voice(user: User, db: DB, file: UploadFile = File(...), limit: int = Form(20)):
        try:
            return db.search_voice(user, await read_limited(file, db), limit=limit)
        except (UnsupportedAudio, ValueError) as e:
            raise HTTPException(422, str(e))

    @app.get("/stats")
    def stats(user: User, db: DB):
        return db.stats(user)

    return app


app = create_app()
