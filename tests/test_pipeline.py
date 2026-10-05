import numpy as np
import pytest

from radiodb.pipeline.diarize import assign_words, build_segments, turns_from_windows, windows
from radiodb.pipeline.identify import best_match
from radiodb.pipeline.interfaces import Span, Turn, Word
from tests.conftest import VOCAB, conversation, script_words, wav_bytes

CONVO = [("A", 4.0), (None, 0.6), ("B", 4.0), (None, 0.6), ("A", 3.0)]


def test_windows_cover_spans_and_skip_tiny_ones():
    wins = windows([Span(0, 3.0), Span(5.0, 5.3)])
    assert wins[0].start == 0 and wins[-1].end == 3.0
    assert all(w.end - w.start >= 0.5 for w in wins)
    assert all(w.start < 5.0 for w in wins)


def test_turns_merge_same_label_windows():
    wins = [Span(0, 1.5), Span(0.75, 2.25), Span(3, 4.5)]
    turns = turns_from_windows(wins, np.array([0, 0, 1]))
    assert [(t.cluster, t.start) for t in turns] == [(0, 0), (1, 3)]
    assert turns[0].end == pytest.approx(2.25)


def test_words_split_on_speaker_change_and_long_pause():
    turns = [Turn(0, 3, 0), Turn(3, 6, 1)]
    words = [Word("a", 0.1, 0.4), Word("b", 0.5, 0.9), Word("c", 2.5, 2.8), Word("d", 3.2, 3.5)]
    segs = build_segments(words, assign_words(words, turns), turns)
    assert [s["text"] for s in segs] == ["a b", "c", "d"]
    assert [s["cluster"] for s in segs] == [0, 0, 1]


def test_zero_length_word_at_boundary_stays_with_previous_speaker():
    turns = [Turn(0, 2.0, 0), Turn(2.0, 4.0, 1)]
    words = [Word("you", 1.5, 1.8), Word("know", 1.8, 2.0), Word("I", 2.0, 2.0), Word("heard", 2.1, 2.4)]
    assert assign_words(words, turns) == [0, 0, 0, 1]


def test_wordless_turns_are_kept():
    turns = [Turn(0, 2, 0), Turn(2.5, 5, 1)]
    words = [Word("hi", 0.2, 0.5)]
    segs = build_segments(words, assign_words(words, turns), turns)
    assert len(segs) == 2 and segs[1]["text"] == "" and segs[1]["cluster"] == 1


def test_small_clusters_are_absorbed_into_nearest_large_one():
    from radiodb.pipeline.diarize import absorb_small_clusters

    wins = windows([Span(0, 6.0), Span(7.0, 7.9), Span(9.0, 14.0)])
    a, b, near_a = np.array([1, 0, 0.0]), np.array([0, 1, 0.0]), np.array([0.9, 0.1, 0.4])
    emb = np.stack([a] * 7 + [near_a] + [b] * 6)
    labels = np.array([0] * 7 + [2] + [1] * 6)
    assert len(emb) == len(wins)
    out = absorb_small_clusters(emb, labels, wins, min_speaker_s=3.0)
    assert set(out) == {0, 1} and out[7] == 0


def test_best_match_respects_threshold():
    a, b = np.array([1, 0], np.float32), np.array([0, 1], np.float32)
    assert best_match(a, [("x", a), ("y", b)], 0.8).speaker_id == "x"
    assert best_match(np.array([0.7, 0.7], np.float32), [("x", a)], 0.8).speaker_id is None


def _process(db, asr, user, script, title="c.wav"):
    asr.next_words = script_words(script, VOCAB)
    rec, _ = db.ingest(user, wav_bytes(conversation(script)), title)
    return db.process(user, rec["id"])


def test_process_diarizes_and_transcribes(db, asr):
    rec = _process(db, asr, "u1", CONVO)
    assert rec["status"] == "done"
    assert "fake-asr" in rec["producers"]

    segs = db.transcript("u1", rec["id"])
    speakers = [s["speaker_id"] for s in segs]
    assert len(set(speakers)) == 2
    assert speakers[0] == speakers[-1] != speakers[1]  # A, B, A
    assert segs[0]["text"].startswith("hello")
    assert segs[1]["text"].startswith("pretty")
    assert segs[1]["start_s"] == pytest.approx(4.6, abs=0.3)


def test_speakers_persist_across_recordings(db, asr):
    _process(db, asr, "u1", CONVO, "one.wav")
    rec2 = _process(db, asr, "u1", [("B", 3.0), (None, 0.6), ("C", 3.0)], "two.wav")
    speakers = db.list_speakers("u1")
    assert len(speakers) == 3  # A, B, and the new C

    b_id = next(s["speaker_id"] for s in db.transcript("u1", rec2["id"]) if s["text"].startswith("pretty"))
    b_segments = db.speaker_segments("u1", b_id)
    assert len({s["recording_id"] for s in b_segments}) == 2


def test_enroll_names_existing_unnamed_speaker(db, asr):
    rec = _process(db, asr, "u1", CONVO)
    assert len(db.list_speakers("u1")) == 2

    alice = db.enroll("u1", "Alice", [wav_bytes(conversation([("A", 5.0)]))])
    assert alice["enrolled"] and alice["name"] == "Alice"
    assert len(db.list_speakers("u1")) == 2  # unnamed A was folded into Alice

    names = [s["speaker"] for s in db.transcript("u1", rec["id"])]
    assert names[0] == names[-1] == "Alice" and names[1] is None

    rec2 = _process(db, asr, "u1", [("A", 3.0)], "later.wav")
    assert {s["speaker"] for s in db.transcript("u1", rec2["id"])} == {"Alice"}


def test_search_text_and_by_speaker(db, asr):
    _process(db, asr, "u1", CONVO)
    db.enroll("u1", "Bob", [wav_bytes(conversation([("B", 5.0)]))])

    hits = db.search_text("u1", "saved")
    assert len(hits) == 1 and hits[0]["speaker"] == "Bob" and "[saved]" in hits[0]["snippet"]
    assert db.search_text("u1", "budget", speaker="bob") == []
    assert db.search_text("u2", "saved") == []
    assert db.search_text("u1", 'odd "query OR') == []  # FTS syntax can't break the query


def test_search_by_voice_identifies_speaker(db, asr):
    _process(db, asr, "u1", CONVO)
    db.enroll("u1", "Bob", [wav_bytes(conversation([("B", 5.0)]))])

    res = db.search_voice("u1", wav_bytes(conversation([("B", 2.0)])))
    assert res["identified"]["name"] == "Bob"
    assert res["segments"][0]["speaker"] == "Bob"

    stranger = db.search_voice("u1", wav_bytes(conversation([("C", 2.0)])))
    assert stranger["identified"] is None


def test_rename_merge_delete_speakers(db, asr):
    rec = _process(db, asr, "u1", CONVO)
    a, b = (s["id"] for s in db.list_speakers("u1"))
    db.rename_speaker("u1", a, "Host")
    with pytest.raises(ValueError):
        db.rename_speaker("u1", b, "host")

    db.merge_speakers("u1", a, b)
    assert {s["speaker"] for s in db.transcript("u1", rec["id"])} == {"Host"}

    db.delete_speaker("u1", a)
    segs = db.catalog.segments("u1", recording_id=rec["id"])
    assert all(s["speaker_id"] is None and s["embedding"] is None for s in segs)
    assert all(s["text"] for s in segs)  # transcript survives, unattributed


def test_failed_processing_is_recorded(db, asr):
    rec, _ = db.ingest("u1", wav_bytes(conversation(CONVO)))
    asr.transcribe = lambda a: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        db.process("u1", rec["id"])
    got = db.get_recording("u1", rec["id"])
    assert got["status"] == "failed" and "boom" in got["error"]


def test_reprocessing_and_deleting_do_not_leave_orphan_speakers(db, asr):
    rec = _process(db, asr, "u1", CONVO)
    db.process("u1", rec["id"])
    assert len(db.list_speakers("u1")) == 2
    db.enroll("u1", "Alice", [wav_bytes(conversation([("A", 5.0)]))])
    db.delete_recording("u1", rec["id"])
    assert [s["name"] for s in db.list_speakers("u1")] == ["Alice"]  # named profiles are kept


def test_min_speaker_scales_down_for_short_recordings():
    from radiodb.pipeline.diarize import effective_min_speaker_s

    short = windows([Span(0, 22.0)])     # ~22 s of speech: a brief phone call
    long_ = windows([Span(0, 1800.0)])   # 30 min meeting
    assert effective_min_speaker_s(short, 20.0) == pytest.approx(2.2, abs=0.05)
    assert effective_min_speaker_s(long_, 20.0) == 20.0
