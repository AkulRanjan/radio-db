"""End-to-end check with the real models on a real two-speaker conversation.

Run with:  pytest -m models
Needs the ``ml`` extras and samples/sample.wav (pyannote's 30 s two-speaker tutorial clip):
  curl -L -o samples/sample.wav https://raw.githubusercontent.com/pyannote/pyannote-audio/main/tutorials/assets/sample.wav
"""

from pathlib import Path

import pytest

from radiodb.config import Settings
from radiodb.service import RadioDB

SAMPLE = Path(__file__).parent.parent / "samples" / "sample.wav"
MODEL_CACHE = Path(__file__).parent.parent / "data" / "models"

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not SAMPLE.exists(), reason="samples/sample.wav not downloaded"),
]


def test_real_conversation(tmp_path, monkeypatch):
    settings = Settings(home=tmp_path)
    monkeypatch.setattr(Settings, "model_dir", property(lambda self: MODEL_CACHE))
    db = RadioDB(settings)

    rec, _ = db.ingest("u1", SAMPLE.read_bytes(), "sample.wav")
    rec = db.process("u1", rec["id"])
    assert rec["status"] == "done", rec["error"]

    segs = db.transcript("u1", rec["id"])
    for s in segs:
        print(f"[{s['start_s']:6.2f}-{s['end_s']:6.2f}] {(s['speaker_id'] or '?')[:6]}: {s['text']}")
    speakers = db.list_speakers("u1")
    print({s["id"][:6]: round(s["talk_time_s"], 1) for s in speakers})

    assert 2 <= len(speakers) <= 3
    assert sum(len(s["text"].split()) for s in segs) > 40

    # Enroll the main speaker from one of their own segments, then identify a different segment.
    main = max(speakers, key=lambda s: s["talk_time_s"])
    theirs = sorted((s for s in segs if s["speaker_id"] == main["id"]), key=lambda s: s["start_s"] - s["end_s"])
    enroll_seg, probe_seg = theirs[0], theirs[1]
    enroll_clip = db.clip("u1", rec["id"], enroll_seg["start_s"], enroll_seg["end_s"])
    db.enroll("u1", "Main", [enroll_clip])

    probe = db.clip("u1", rec["id"], probe_seg["start_s"], probe_seg["end_s"])
    res = db.search_voice("u1", probe)
    print("voice search:", [(s["name"], s["score"]) for s in res["speakers"]])
    assert res["identified"] and res["identified"]["name"] == "Main"

    hits = db.search_text("u1", segs[0]["text"].split()[0])
    assert hits
    db.catalog.close()

