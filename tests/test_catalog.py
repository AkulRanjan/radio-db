"""Catalog behaviour that must hold on every backend (SQLite always; Postgres when configured)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from radiodb.catalog.base import SchemaVersionError
from tests.conftest import VOCAB, conversation, script_words, wav_bytes

V = np.ones(4, dtype=np.float32) / 2


def test_transaction_rolls_back_on_error(catalog):
    with pytest.raises(RuntimeError):
        with catalog.transaction():
            catalog.add_speaker("u1", V, name="ghost")
            raise RuntimeError("boom")
    assert catalog.list_speakers("u1") == []


def test_nested_transaction_is_atomic(catalog):
    with catalog.transaction():
        catalog.add_speaker("u1", V, name="outer")
        with catalog.transaction():
            catalog.add_speaker("u1", V, name="inner")
    assert {s["name"] for s in catalog.list_speakers("u1")} == {"outer", "inner"}


def test_refuses_newer_schema(catalog):
    catalog._x("UPDATE radiodb_meta SET value='99' WHERE key='schema_version'")
    with pytest.raises(SchemaVersionError, match="newer"):
        catalog._check_version()


def test_speaker_name_lookup_is_case_insensitive(catalog):
    catalog.add_speaker("u1", V, name="Alice")
    assert catalog.find_speaker_by_name("u1", "ALICE")["name"] == "Alice"
    assert catalog.find_speaker_by_name("u2", "alice") is None


def test_timestamps_round_trip_as_iso_strings(catalog):
    rec = catalog.add_recording(user_id="u1", title=None, sha256="x" * 64, blob_key="k", format="flac",
                                sample_rate=16000, channels=1, duration_s=1.0, retention_days=365)
    assert isinstance(rec["created_at"], str) and rec["created_at"].endswith("+00:00")
    assert rec["expires_at"] > rec["created_at"]


def test_concurrent_requests(db, asr):
    """Ingest, process and search from many threads at once (as API workers would)."""
    def work(i):
        script = [("A", 2.0 + i * 0.1), (None, 0.6), ("B", 2.0)]
        rec, _ = db.ingest(f"user{i % 3}", wav_bytes(conversation(script)), f"r{i}.wav")
        db.search(f"user{i % 3}", "budget")
        db.list_speakers(f"user{i % 3}")
        return rec["id"]

    asr.next_words = script_words([("A", 2.0), (None, 0.6), ("B", 2.0)], VOCAB)
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(work, range(12)))
    assert len(set(ids)) == 12
    assert sum(len(db.list_recordings(f"user{u}")) for u in range(3)) == 12
    for i, rid in enumerate(ids[:3]):  # processing (GPU-serialised) works after concurrent ingest
        assert db.process(f"user{i % 3}", rid)["status"] == "done"
