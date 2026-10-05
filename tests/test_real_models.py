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
EMOTION_LOCAL = MODEL_CACHE / "emotion" / "superb-hubert-base-superb-er"
# Use a locally downloaded emotion model when present (the hub download is unreliable here).
EMOTION = {"emotion_model": str(EMOTION_LOCAL)} if (EMOTION_LOCAL / "pytorch_model.bin").exists() else {}

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not SAMPLE.exists(), reason="samples/sample.wav not downloaded"),
]


def test_real_conversation(tmp_path, monkeypatch):
    settings = Settings(home=tmp_path, **EMOTION)
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

    assert len(speakers) == 2
    full = " ".join(s["text"] for s in segs)
    assert len(full.split()) > 40
    # Regression: torch leaving flush-to-zero on after ECAPA used to turn this into "of deep".
    assert "heard a beep" in full

    # Enrichment layers ran on real audio (or recorded why not).
    assert any(s["prosody"] and s["prosody"]["f0_median_hz"] for s in segs)
    if EMOTION:
        assert all(s["emotion"] for s in segs if s["end_s"] - s["start_s"] >= 1.0)
    where = db.search("u1", "where did they grow up", mode="semantic", limit=2)
    assert where and all("Chicago" in h["text"] for h in where)

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



def test_long_lived_worker_keeps_transcripts_stable(tmp_path, monkeypatch):
    """Regression: in one long-lived worker, ECAPA (torch/oneDNN) used to leave shared OpenMP
    threads in flush-to-zero mode and degrade later CPU Whisper runs (5/5 runs, measured).
    ASR now runs in an isolated process on CPU."""
    from radiodb.worker import Worker

    monkeypatch.setattr(Settings, "model_dir", property(lambda self: MODEL_CACHE))
    db = RadioDB(Settings(home=tmp_path, device="cpu", layers=()))
    data = SAMPLE.read_bytes()
    for i in range(2):
        rec, _ = db.ingest(f"u{i}", data)
        db.enqueue_process(f"u{i}", rec["id"])
        assert Worker(db).run_until_idle() == 1
        text = " ".join(s["text"] for s in db.transcript(f"u{i}", rec["id"]))
        assert "heard a beep" in text, f"job {i + 1}: {text[:120]}"
    db.models.transcriber.close()
    db.catalog.close()



@pytest.mark.skipif(not (EMOTION_LOCAL / "pytorch_model.bin").exists(), reason="emotion model not downloaded")
def test_emotion_layer_on_real_audio(tmp_path, monkeypatch):
    from radiodb.pipeline.layers import EmotionLayer, HubertEmotionClassifier
    from radiodb import audio

    clf = HubertEmotionClassifier(str(EMOTION_LOCAL), MODEL_CACHE, "cpu")
    assert clf.labels == ["neutral", "happy", "angry", "sad"] and clf.name == "local:superb-hubert-base-superb-er"
    a16 = audio.to_model_input(audio.decode(SAMPLE.read_bytes()))
    rows = EmotionLayer(clf).compute(a16, [{"start_s": 13.56, "end_s": 17.54}, {"start_s": 10.08, "end_s": 10.64}])
    calm, short = rows[0][0], rows[1][0]
    assert calm["label"] == "neutral" and abs(sum(calm["probs"].values()) - 1) < 0.01
    assert short is None  # 0.56 s: below the 1 s minimum
