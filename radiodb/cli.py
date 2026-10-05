"""Command-line interface: ``radiodb --help``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .service import RadioDB

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Store and search conversational audio.")
speakers_app = typer.Typer(no_args_is_help=True, help="Manage speaker identities.")
app.add_typer(speakers_app, name="speakers")

UserOpt = typer.Option("local", "--user", "-u", envvar="RADIODB_USER", help="Owner of the data.")


def _db() -> RadioDB:
    return RadioDB()


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _fmt_t(s: float) -> str:
    m, s = divmod(s, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}"


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, dir_okay=False),
    user: str = UserOpt,
    process: bool = typer.Option(True, help="Process the audio (diarize, identify, transcribe, layers)."),
    queue: bool = typer.Option(False, "--queue", help="Queue processing for `radiodb worker` instead of running it now."),
):
    """Store audio files (and process them)."""
    db = _db()
    for p in paths:
        rec, created = db.ingest(user, p.read_bytes(), title=p.name)
        typer.echo(f"{'stored ' if created else 'exists '} {rec['id']}  {p.name}  ({rec['duration_s']:.1f}s)")
        if process and (created or rec["status"] != "done"):
            if queue:
                typer.echo(f"         queued: job {db.enqueue_process(user, rec['id'])['id']}")
            else:
                rec = db.process(user, rec["id"])
                typer.echo(f"         processed: {rec['status']}")


@app.command()
def process(rec_id: str, user: str = UserOpt):
    """(Re)run everything for a recording: diarization, identification, ASR and layers."""
    _print(_db().process(user, rec_id))


@app.command("list")
def list_(user: str = UserOpt):
    """List recordings."""
    for r in _db().list_recordings(user):
        typer.echo(f"{r['id']}  {r['status']:<10} {r['duration_s']:>8.1f}s  expires {r['expires_at'][:10]}  {r['title'] or ''}")


def _tags(seg: dict) -> str:
    st = seg.get("style") or {}
    tags = [st[d] for d in ("pitch", "pace", "volume", "expressiveness") if st.get(d) not in (None, "typical")]
    emo = seg.get("emotion")
    if emo and emo["confident"] and emo["label"] != "neutral":
        tags.append(f"{emo['label']} {emo['confidence']:.0%}")
    return f"  ({', '.join(tags)})" if tags else ""


@app.command()
def show(rec_id: str, user: str = UserOpt):
    """Print a speaker-attributed transcript, tagged with anything unusual about how it was said."""
    for s in _db().transcript(user, rec_id):
        who = s["speaker"] or (s["speaker_id"] or "?")[:8]
        typer.echo(f"[{_fmt_t(s['start_s'])} - {_fmt_t(s['end_s'])}] {who}: {s['text'] or '(no words)'}{_tags(s)}")


@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Words or a description. Omit to filter by style only."),
    user: str = UserOpt,
    mode: str = typer.Option("hybrid", help="hybrid | keyword | semantic"),
    speaker: Optional[str] = None,
    recording: Optional[str] = None,
    pitch: Optional[str] = typer.Option(None, help="low | typical | high (relative to the speaker)"),
    pace: Optional[str] = typer.Option(None, help="slow | typical | fast"),
    volume: Optional[str] = typer.Option(None, help="quiet | typical | loud"),
    expressiveness: Optional[str] = typer.Option(None, help="flat | typical | animated"),
    emotion: Optional[str] = typer.Option(None, help="neutral | happy | angry | sad (heard in the voice)"),
    limit: int = 20,
):
    """Search what was said (by words and/or meaning) and how it was said."""
    style = {k: v for k, v in
             {"pitch": pitch, "pace": pace, "volume": volume, "expressiveness": expressiveness}.items() if v}
    hits = _db().search(user, query, mode=mode, speaker=speaker, recording=recording, style=style,
                        emotion=emotion, limit=limit)
    for h in hits:
        who = h["speaker"] or (h["speaker_id"] or "?")[:8]
        text = h.get("snippet") or h["text"] or "(no words)"
        typer.echo(f"{h['recording_id']} [{_fmt_t(h['start_s'])}] {who}: {text}{_tags(h)}")


@app.command()
def enrich(
    rec_id: Optional[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Every processed recording of the user."),
    layer: Optional[list[str]] = typer.Option(None, "--layer", "-l", help="Layer(s) to build; default all enabled."),
    user: str = UserOpt,
):
    """Build derived layers (prosody, semantic) on processed recordings without re-running ASR."""
    db = _db()
    if all_:
        ids = [r["id"] for r in db.list_recordings(user) if r["status"] == "done"]
    elif rec_id:
        ids = [rec_id]
    else:
        raise typer.BadParameter("give a recording id or --all")
    for rid in ids:
        db.enrich(user, rid, layer or None)
        typer.echo(f"enriched {rid}")


@app.command("search-voice")
def search_voice(clip: Path = typer.Argument(..., exists=True), user: str = UserOpt, limit: int = 10):
    """Who is speaking in this clip, and where else do they appear?"""
    _print(_db().search_voice(user, clip.read_bytes(), limit=limit))


@app.command()
def clip(
    rec_id: str, start: float, end: float, out: Path = typer.Option(..., "--out", "-o"), user: str = UserOpt
):
    """Export a time range of a recording as WAV."""
    out.write_bytes(_db().clip(user, rec_id, start, end))
    typer.echo(f"wrote {out}")


@app.command()
def delete(rec_id: str, user: str = UserOpt):
    """Delete a recording and everything derived from it."""
    _db().delete_recording(user, rec_id)
    typer.echo("deleted")


@app.command()
def purge():
    """Delete all recordings past their retention date (schedule daily)."""
    typer.echo(f"purged {_db().purge_expired()} recording(s)")


@app.command()
def rttm(rec_id: str, out: Optional[Path] = typer.Option(None, "--out", "-o"), user: str = UserOpt):
    """Export a recording's diarization as RTTM: a starting point for labelling calibration data."""
    from .calibrate import to_rttm

    segs = _db().transcript(user, rec_id)
    names = {}
    rows = [(s["start_s"], s["end_s"],
             (s["speaker"] or names.setdefault(s["speaker_id"], f"speaker{len(names) + 1}")).replace(" ", "_"))
            for s in segs if s["speaker_id"]]
    text = to_rttm(rec_id, rows)
    if out:
        out.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {out} ({len(rows)} segments)")
    else:
        typer.echo(text, nl=False)


@app.command()
def calibrate(
    directory: Path = typer.Argument(..., exists=True, file_okay=False,
                                     help="Audio files + same-named .rttm (+ optional .txt reference transcripts)."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write the full JSON report here."),
    no_asr: bool = typer.Option(False, "--no-asr", help="Skip ASR word-error-rate measurement."),
    collar: float = typer.Option(0.25, help="Seconds around reference boundaries not scored for DER."),
    thresholds: str = typer.Option("0.5,0.6,0.7,0.75,0.8,0.85,0.9", help="Diarization thresholds to sweep."),
    min_speaker: str = typer.Option("3,10,20,30,60", help="Minimum speech per speaker (s) to sweep."),
):
    """Measure diarization, speaker identification and ASR on labelled data; recommend settings."""
    from .calibrate import calibrate as run

    db = _db()
    typer.echo(f"calibrating on {directory} ...")
    rep = run(directory, db.models, collar=collar, run_asr=not no_asr, progress=typer.echo,
              thresholds=tuple(float(x) for x in thresholds.split(",")),
              min_speaker_s=tuple(float(x) for x in min_speaker.split(",")))
    if out:
        out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        typer.echo(f"full report: {out}")

    b, ident, s = rep["diarization"]["best"], rep["identification"], db.settings
    typer.echo()
    typer.echo(f"{len(rep['files'])} file(s), {rep['total_audio_s'] / 60:.1f} min of audio")
    typer.echo(f"diarization  best DER {b['der']:.1%} (miss {b['miss']:.1%}, false alarm {b['false_alarm']:.1%}, "
               f"confusion {b['confusion']:.1%}) at threshold {b['diarization_threshold']}, "
               f"min speaker {b['min_speaker_s']} s")
    current = next((g for g in rep["diarization"]["grid"] if g["diarization_threshold"] == s.diarization_threshold
                    and g["min_speaker_s"] == s.min_speaker_s), None)
    if current:
        typer.echo(f"             current settings DER {current['der']:.1%}")
    if "eer" in ident:
        typer.echo(f"identification  EER {ident['eer']:.1%} over {ident['genuine_pairs']} same / "
                   f"{ident['impostor_pairs']} different-speaker pairs; recommended threshold "
                   f"{ident['recommended_threshold']} (current {s.speaker_match_threshold})")
    else:
        typer.echo(f"identification  {ident['note']}")
    if rep["asr"]:
        typer.echo(f"asr  WER {rep['asr']['wer']:.1%} over {rep['asr']['ref_words']} words")
    rtf = rep["real_time_factor"]
    typer.echo("speed  " + ", ".join(f"{k} {v:.2f}x real time" for k, v in rtf.items() if v is not None))
    typer.echo()
    typer.echo("recommended settings:")
    for k, v in rep["recommended_env"].items():
        typer.echo(f"  {k}={v}")


@app.command()
def stats(user: str = UserOpt):
    _print(_db().stats(user))


@app.command()
def serve(
    host: str = "127.0.0.1", port: int = 8000,
    worker: bool = typer.Option(True, help="Also run a job worker inside the API process (local dev)."),
):
    """Run the HTTP API."""
    import os

    import uvicorn

    os.environ["RADIODB_INLINE_WORKER"] = "1" if worker else "0"
    uvicorn.run("radiodb.api:app", host=host, port=port)


@app.command("worker")
def worker_cmd(
    once: bool = typer.Option(False, "--once", help="Run the jobs that are due now, then exit."),
    lease: float = typer.Option(600, help="Seconds a claimed job is leased before others may take it over."),
    poll: float = typer.Option(2.0, help="Seconds between polls when the queue is empty."),
    kind: Optional[list[str]] = typer.Option(None, "--kind", help="Only run these job kinds (process, enrich, purge)."),
):
    """Run a background worker that processes queued jobs (run one or more in production)."""
    import logging
    import signal

    from .worker import Worker

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "speechbrain", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    w = Worker(_db(), lease_s=lease, heartbeat_s=max(5.0, lease / 10), poll_s=poll, kinds=kind or None)
    if once:
        typer.echo(f"ran {w.run_until_idle()} job(s)")
        return
    signal.signal(signal.SIGINT, lambda *_: w.stop())
    signal.signal(signal.SIGTERM, lambda *_: w.stop())
    w.run()


@app.command()
def jobs(user: str = UserOpt, status: Optional[str] = None, limit: int = 20):
    """List queued / running / finished jobs."""
    for j in _db().list_jobs(user, status, limit):
        err = f"  {j['last_error']}" if j["last_error"] else ""
        typer.echo(f"{j['id']}  {j['kind']:<8} {j['status']:<8} try {j['attempts']}/{j['max_attempts']}"
                   f"  {j['recording_id'] or '-'}{err}")


@speakers_app.command("list")
def speakers_list(user: str = UserOpt):
    for s in _db().list_speakers(user):
        typer.echo(
            f"{s['id']}  {(s['name'] or '(unnamed)'):<20} {'enrolled' if s['enrolled'] else '        '}"
            f"  {s['n_segments']:>5} segs  {s['talk_time_s'] / 60:>7.1f} min"
        )


@speakers_app.command("profile")
def speakers_profile(speaker: str, user: str = UserOpt):
    """How a speaker usually sounds (the baseline their style labels are relative to)."""
    _print(_db().speaker_profile(user, speaker))


@speakers_app.command("enroll")
def speakers_enroll(name: str, samples: list[Path] = typer.Argument(..., exists=True), user: str = UserOpt):
    """Create a named voice profile from sample clips (10-30 s of clean speech is ideal)."""
    _print(_db().enroll(user, name, [p.read_bytes() for p in samples]))


@speakers_app.command("rename")
def speakers_rename(speaker_id: str, name: str, user: str = UserOpt):
    _print(_db().rename_speaker(user, speaker_id, name))


@speakers_app.command("merge")
def speakers_merge(keep: str, drop: str, user: str = UserOpt):
    """Attribute everything said by DROP to KEEP."""
    _print(_db().merge_speakers(user, keep, drop))


@speakers_app.command("delete")
def speakers_delete(speaker_id: str, user: str = UserOpt):
    """Delete a voice profile and its voice embeddings (transcripts are kept, unattributed)."""
    _db().delete_speaker(user, speaker_id)
    typer.echo("deleted")


if __name__ == "__main__":
    app()
