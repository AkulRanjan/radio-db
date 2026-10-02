"""HTTP API. Local dev identifies the user with the ``X-User-Id`` header; production must put
real authentication in front of this (the header should come from a verified token)."""


from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .audio import UnsupportedAudio
from .service import NotFound, RadioDB

User = Annotated[str, Header(alias="X-User-Id", min_length=1, max_length=128)]

MEDIA_TYPES = {"flac": "audio/flac", "wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}


class RenameBody(BaseModel):
    name: str


class MergeBody(BaseModel):
    keep: str
    drop: str


def create_app(db: RadioDB | None = None) -> FastAPI:
    app = FastAPI(title="radio-db", version="0.1.0")
    state = {"db": db}

    def get_db() -> RadioDB:
        if state["db"] is None:
            state["db"] = RadioDB()
        return state["db"]

    DB = Annotated[RadioDB, Depends(get_db)]

    @app.exception_handler(NotFound)
    async def _nf(_req, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})

    def _process_safely(db: RadioDB, user: str, rec_id: str) -> None:
        try:
            db.process(user, rec_id)
        except Exception:  # status/error is recorded on the recording row
            pass

    @app.post("/recordings", status_code=201)
    async def upload(
        user: User, db: DB, tasks: BackgroundTasks,
        file: UploadFile = File(...), title: str | None = Form(None), process: bool = Form(True),
    ):
        data = await file.read()
        try:
            rec, created = db.ingest(user, data, title or file.filename)
        except UnsupportedAudio as e:
            raise HTTPException(415, str(e))
        if process and (created or rec["status"] == "failed"):
            tasks.add_task(_process_safely, db, user, rec["id"])
        return {"recording": rec, "created": created}

    @app.get("/recordings")
    def list_recordings(user: User, db: DB):
        return db.list_recordings(user)

    @app.get("/recordings/{rec_id}")
    def get_recording(user: User, db: DB, rec_id: str):
        return db.get_recording(user, rec_id)

    @app.post("/recordings/{rec_id}/process")
    def reprocess(user: User, db: DB, rec_id: str, tasks: BackgroundTasks):
        db.get_recording(user, rec_id)
        tasks.add_task(_process_safely, db, user, rec_id)
        return {"queued": rec_id}

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
            return db.enroll(user, name, [await f.read() for f in files])
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

    @app.get("/speakers/{ref}/segments")
    def speaker_segments(user: User, db: DB, ref: str):
        return db.speaker_segments(user, ref)

    @app.get("/search")
    def search(
        user: User, db: DB, q: str = Query(..., min_length=1), speaker: str | None = None,
        limit: int = Query(20, ge=1, le=200),
    ):
        return db.search_text(user, q, speaker=speaker, limit=limit)

    @app.post("/search/voice")
    async def search_voice(user: User, db: DB, file: UploadFile = File(...), limit: int = Form(20)):
        try:
            return db.search_voice(user, await file.read(), limit=limit)
        except (UnsupportedAudio, ValueError) as e:
            raise HTTPException(422, str(e))

    @app.get("/stats")
    def stats(user: User, db: DB):
        return db.stats(user)

    return app


app = create_app()
