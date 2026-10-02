import io
from datetime import timedelta

import numpy as np
import pytest
import soundfile as sf

from radiodb import audio
from radiodb.catalog import _iso, now_utc
from tests.conftest import conversation, wav_bytes

SCRIPT = [("A", 3.0), (None, 0.5), ("B", 3.0)]


def test_wav_is_stored_as_lossless_flac(db):
    samples = conversation(SCRIPT, sr=44_100)
    rec, created = db.ingest("u1", wav_bytes(samples, sr=44_100), "talk.wav")
    assert created and rec["format"] == "flac" and rec["sample_rate"] == 44_100
    assert rec["duration_s"] == pytest.approx(6.5, abs=0.01)

    stored = db.blobs.get(rec["blob_key"])
    back, sr = sf.read(io.BytesIO(stored), dtype="float32")
    original, _ = sf.read(io.BytesIO(wav_bytes(samples, sr=44_100)), dtype="float32")
    assert sr == 44_100
    np.testing.assert_array_equal(back, original)  # bit-exact round trip


def test_lossy_upload_is_kept_byte_for_byte(db):
    data = wav_bytes(conversation(SCRIPT), fmt="OGG", subtype="VORBIS")
    rec, _ = db.ingest("u1", data)
    assert rec["format"] == "ogg"
    assert db.blobs.get(rec["blob_key"]) == data


def test_ingest_dedupes_per_user(db):
    data = wav_bytes(conversation(SCRIPT))
    r1, c1 = db.ingest("u1", data)
    r2, c2 = db.ingest("u1", data)
    r3, c3 = db.ingest("u2", data)
    assert (c1, c2, c3) == (True, False, True)
    assert r1["id"] == r2["id"] != r3["id"]
    assert r1["blob_key"] == r3["blob_key"]  # one physical blob


def test_rejects_non_audio(db):
    with pytest.raises(audio.UnsupportedAudio):
        db.ingest("u1", b"not audio at all")


def test_retention_defaults_to_one_year(db):
    rec, _ = db.ingest("u1", wav_bytes(conversation(SCRIPT)))
    created = audio_dt(rec["created_at"])
    assert audio_dt(rec["expires_at"]) - created == timedelta(days=365)


def test_purge_removes_expired_rows_and_unshared_blobs(db):
    keep, _ = db.ingest("u1", wav_bytes(conversation(SCRIPT)))
    old, _ = db.ingest("u1", wav_bytes(conversation([("B", 2.0)])))
    db.catalog._x("UPDATE recordings SET expires_at=? WHERE id=?", (_iso(now_utc() - timedelta(days=1)), old["id"]))

    assert db.purge_expired() == 1
    assert db.catalog.get_recording("u1", old["id"]) is None
    assert not db.blobs.exists(old["blob_key"])
    assert db.blobs.exists(keep["blob_key"])


def test_shared_blob_survives_other_users_delete(db):
    data = wav_bytes(conversation(SCRIPT))
    r1, _ = db.ingest("u1", data)
    db.ingest("u2", data)
    db.delete_recording("u1", r1["id"])
    assert db.blobs.exists(r1["blob_key"])


def test_clip_returns_requested_range(db):
    rec, _ = db.ingest("u1", wav_bytes(conversation(SCRIPT)))
    clip, sr = sf.read(io.BytesIO(db.clip("u1", rec["id"], 1.0, 2.5)))
    assert len(clip) / sr == pytest.approx(1.5, abs=1e-3)


def test_users_cannot_see_each_others_recordings(db):
    rec, _ = db.ingest("u1", wav_bytes(conversation(SCRIPT)))
    from radiodb.service import NotFound

    with pytest.raises(NotFound):
        db.get_recording("u2", rec["id"])
    assert db.list_recordings("u2") == []


def audio_dt(s):
    from datetime import datetime

    return datetime.fromisoformat(s)
