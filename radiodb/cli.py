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
    process: bool = typer.Option(True, help="Run diarization + transcription right away."),
):
    """Store audio files (and process them)."""
    db = _db()
    for p in paths:
        rec, created = db.ingest(user, p.read_bytes(), title=p.name)
        typer.echo(f"{'stored ' if created else 'exists '} {rec['id']}  {p.name}  ({rec['duration_s']:.1f}s)")
        if process and (created or rec["status"] != "done"):
            rec = db.process(user, rec["id"])
            typer.echo(f"         processed: {rec['status']}")


@app.command()
def process(rec_id: str, user: str = UserOpt):
    """(Re)build derived layers for a recording."""
    _print(_db().process(user, rec_id))


@app.command("list")
def list_(user: str = UserOpt):
    """List recordings."""
    for r in _db().list_recordings(user):
        typer.echo(f"{r['id']}  {r['status']:<10} {r['duration_s']:>8.1f}s  expires {r['expires_at'][:10]}  {r['title'] or ''}")


@app.command()
def show(rec_id: str, user: str = UserOpt):
    """Print a speaker-attributed transcript."""
    for s in _db().transcript(user, rec_id):
        who = s["speaker"] or (s["speaker_id"] or "?")[:8]
        typer.echo(f"[{_fmt_t(s['start_s'])} - {_fmt_t(s['end_s'])}] {who}: {s['text'] or '(no words)'}")


@app.command()
def search(query: str, user: str = UserOpt, speaker: Optional[str] = None, limit: int = 20):
    """Full-text search over what was said."""
    for h in _db().search_text(user, query, speaker=speaker, limit=limit):
        who = h["speaker"] or (h["speaker_id"] or "?")[:8]
        typer.echo(f"{h['recording_id']} [{_fmt_t(h['start_s'])}] {who}: {h['snippet']}")


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
def stats(user: str = UserOpt):
    _print(_db().stats(user))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("radiodb.api:app", host=host, port=port)


@speakers_app.command("list")
def speakers_list(user: str = UserOpt):
    for s in _db().list_speakers(user):
        typer.echo(
            f"{s['id']}  {(s['name'] or '(unnamed)'):<20} {'enrolled' if s['enrolled'] else '        '}"
            f"  {s['n_segments']:>5} segs  {s['talk_time_s'] / 60:>7.1f} min"
        )


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
