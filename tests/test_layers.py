import numpy as np
import pytest

from radiodb.pipeline.layers import count_syllables, timing_features
from radiodb.query import rrf, style_table
from tests.conftest import SR, VOCAB, Models, conversation, script_words, tone, wav_bytes

parselmouth = pytest.importorskip("parselmouth")


# ---- units ---------------------------------------------------------------------------------

@pytest.mark.parametrize("word,n", [("budget", 2), ("hello", 2), ("the", 1), ("money", 2), ("today", 2),
                                    ("conversation", 4), ("I'm", 1), ("...", 0)])
def test_count_syllables(word, n):
    assert count_syllables(word) == n


def test_timing_features_rates_and_pauses():
    words = [{"w": "hello", "start": 0.0, "end": 0.4}, {"w": "there", "start": 0.45, "end": 0.8},
             {"w": "budget", "start": 1.8, "end": 2.2}]
    f = timing_features(words, 0.0, 2.2)
    assert f["n_words"] == 3 and f["n_pauses"] == 1
    assert f["pause_total_s"] == pytest.approx(1.0)
    assert f["speech_rate_sps"] == pytest.approx(5 / 2.2, abs=1e-3)       # 2 + 1 + 2 syllables
    assert f["articulation_rate_sps"] == pytest.approx(5 / 1.2, abs=1e-3)  # pause removed


def test_prosody_layer_measures_pitch_and_loudness():
    from radiodb.pipeline.layers import ProsodyLayer

    a = np.concatenate([tone(150, 2.0, amp=0.05), tone(300, 2.0, amp=0.5)])
    segs = [{"start_s": 0.0, "end_s": 2.0, "words": []}, {"start_s": 2.0, "end_s": 4.0, "words": []}]
    low, high = (d for d, _ in ProsodyLayer().compute(a, segs))
    assert low["f0_median_hz"] == pytest.approx(150, rel=0.02)
    assert high["f0_median_hz"] == pytest.approx(300, rel=0.02)
    assert high["f0_mean_st"] - low["f0_mean_st"] == pytest.approx(12, abs=0.3)  # one octave
    assert high["intensity_db"] - low["intensity_db"] == pytest.approx(20, abs=1)  # 10x amplitude
    assert low["voiced_frac"] > 0.9


def test_style_is_relative_to_each_speaker():
    # Speaker x talks slowly by nature, speaker y fast; each gets one outlier segment.
    segs, pros = [], {}
    for i, (spk, rate) in enumerate([("x", r) for r in (3.0, 3.1, 2.9, 3.0, 3.2, 5.0)] +
                                    [("y", r) for r in (6.0, 6.1, 5.9, 6.0, 6.2, 4.0)]):
        segs.append({"id": i, "speaker_id": spk, "recording_id": "r"})
        pros[i] = {"articulation_rate_sps": rate}
    st = style_table(segs, pros)
    assert st[5]["pace"] == "fast" and st[11]["pace"] == "slow"  # same-ish absolute rate (5 vs 4)…
    assert st[0]["pace"] == st[6]["pace"] == "typical"            # …but judged per speaker


def test_style_needs_a_baseline():
    segs = [{"id": i, "speaker_id": "x", "recording_id": "r"} for i in range(3)]
    assert style_table(segs, {i: {"articulation_rate_sps": 9.0 * i} for i in range(3)}) == {}


def test_rrf_rewards_agreement():
    s = rrf([1, 2, 3], [3, 1])
    assert max(s, key=s.get) == 1 and s[3] > s[2]


# ---- through the service -------------------------------------------------------------------

# Speaker A has six turns; the fifth is shouted. B answers in between.
LOUD_TURN = 4
A_TURNS = [("A", 5.0, 0.9 if i == LOUD_TURN else 0.1) for i in range(6)]
SCRIPT = [x for a in A_TURNS for x in (a, (None, 0.6), ("B", 3.0, 0.3), (None, 0.6))]


def _process(db, asr, script=SCRIPT, user="u1"):
    asr.next_words = script_words(script, VOCAB)
    rec, _ = db.ingest(user, wav_bytes(conversation(script)), "talk.wav")
    return db.process(user, rec["id"])


def test_layers_are_built_with_provenance(db, asr):
    rec = _process(db, asr)
    assert set(json_loads(rec["producers"])["layers"]) == {"prosody", "semantic", "emotion"}
    seg = db.transcript("u1", rec["id"])[0]
    assert seg["prosody"]["f0_median_hz"] == pytest.approx(220, rel=0.02)
    assert seg["style"]["volume"] == "typical"


def test_find_the_shouted_turn_by_style(db, asr):
    rec = _process(db, asr)
    loud = db.search("u1", style={"volume": "loud"})
    assert len(loud) == 1
    a_turns = [s for s in db.transcript("u1", rec["id"]) if s["text"].startswith("hello")]
    assert loud[0]["id"] == a_turns[LOUD_TURN]["id"]
    with pytest.raises(ValueError, match="invalid style"):
        db.search("u1", style={"volume": "deafening"})


def test_semantic_search_finds_meaning_without_shared_words(db, asr):
    _process(db, asr)
    assert db.search("u1", "finances", mode="keyword") == []
    hits = db.search("u1", "finances", mode="semantic")
    direct = [bool({"budget", "money", "saved"} & set(h["text"].split())) for h in hits]
    assert direct[0] and direct == sorted(direct, reverse=True)  # own words outrank context
    assert db.search("u1", "running late", mode="semantic") == []  # no false matches

    # "pretty" only matches B by keyword; "finances" only matches A by meaning. Hybrid needs both.
    hybrid = db.search("u1", "pretty finances", limit=50)
    starts = {h["text"].split()[0] for h in hybrid}
    assert {"pretty", "hello"} <= starts
    assert any("snippet" in h for h in hybrid if h["text"].startswith("pretty"))


def test_semantic_search_respects_speaker_filter(db, asr):
    _process(db, asr)
    a = db.list_speakers("u1")[0]["id"]
    assert {h["speaker_id"] for h in db.search("u1", "finances", speaker=a)} == {a}


def test_enrich_backfills_layers_without_reprocessing(db, asr):
    layers = db.models.layers
    db.models.layers = []
    rec = _process(db, asr)
    assert db.transcript("u1", rec["id"])[0]["prosody"] is None
    seg_ids = [s["id"] for s in db.transcript("u1", rec["id"])]

    db.models.layers = layers
    asr.transcribe = lambda a: pytest.fail("enrich must not re-run ASR")
    rec = db.enrich("u1", rec["id"], ["prosody"])
    assert json_loads(rec["producers"])["layers"] == {"prosody": layers[0].producer}
    segs = db.transcript("u1", rec["id"])
    assert [s["id"] for s in segs] == seg_ids and segs[0]["prosody"] is not None
    with pytest.raises(ValueError, match="unknown"):
        db.enrich("u1", rec["id"], ["telepathy"])


def test_speaker_profile_reports_baseline(db, asr):
    _process(db, asr)
    db.enroll("u1", "Alice", [wav_bytes(conversation([("A", 5.0)]))])
    prof = db.speaker_profile("u1", "alice")
    assert prof["voice"]["f0_median_hz"] == pytest.approx(220, rel=0.02)
    assert prof["segments"] == 6 and prof["recordings"] == 1


def test_deleting_speaker_removes_voice_derived_features(db, asr):
    rec = _process(db, asr)
    a = db.list_speakers("u1")[0]["id"]
    db.delete_speaker("u1", a)
    segs = [s for s in db.transcript("u1", rec["id"]) if s["text"].startswith("hello")]
    assert segs and all(s["prosody"] is None and s["speaker_id"] is None for s in segs)


def json_loads(s):
    import json

    return json.loads(s)


# ---- emotion -------------------------------------------------------------------------------

def test_emotion_windows_are_length_weighted():
    from radiodb.pipeline.layers import EmotionLayer
    from tests.conftest import FakeEmotionClassifier

    a = np.concatenate([tone(220, 10.0, amp=0.9), tone(220, 4.0, amp=0.2)])  # window 1 shouting, window 2 calm
    (d, _), (short, _) = EmotionLayer(FakeEmotionClassifier()).compute(
        a, [{"start_s": 0.0, "end_s": 14.0}, {"start_s": 0.0, "end_s": 0.5}])
    assert d["label"] == "angry"
    assert d["probs"]["angry"] == pytest.approx((10 * 0.8 + 4 * 0.1) / 14, abs=1e-3)
    assert short is None  # under 1 s: not scored


def test_search_by_emotion_and_confidence(db, asr):
    rec = _process(db, asr)
    angry = db.search("u1", emotion="angry")
    a_turns = [s for s in db.transcript("u1", rec["id"]) if s["text"].startswith("hello")]
    assert [h["id"] for h in angry] == [a_turns[LOUD_TURN]["id"]]
    assert angry[0]["emotion"]["confident"] and angry[0]["emotion"]["label"] == "angry"

    object.__setattr__(db.settings, "emotion_min_confidence", 0.9)  # stricter than the 0.8 we produce
    assert db.search("u1", emotion="angry") == []


def test_speaker_profile_emotion_mix(db, asr):
    _process(db, asr)
    db.enroll("u1", "Alice", [wav_bytes(conversation([("A", 5.0)]))])
    emo = db.speaker_profile("u1", "alice")["emotion"]
    assert emo["scored_segments"] == 6
    assert emo["share"] == {"angry": round(1 / 6, 3), "neutral": round(5 / 6, 3)}


def test_deleting_speaker_removes_emotion_too(db, asr):
    rec = _process(db, asr)
    a = db.list_speakers("u1")[0]["id"]
    db.delete_speaker("u1", a)
    assert db.catalog.layers("u1", "emotion", speaker_id=a) == {}
    a_segs = [s for s in db.transcript("u1", rec["id"]) if s["text"].startswith("hello")]
    assert all(s["emotion"] is None for s in a_segs)
    others = [s for s in db.transcript("u1", rec["id"]) if s["text"].startswith("pretty")]
    assert all(s["emotion"] is not None for s in others)  # other speakers untouched


# ---- graceful degradation -------------------------------------------------------------------

class _BrokenLayer:
    name, producer = "emotion", "broken-model"

    def compute(self, a16, segs):
        raise RuntimeError("out of memory")


def test_failing_layer_keeps_core_results_and_other_layers(db, asr):
    good = db.models.layers
    db.models.layers = [l for l in good if l.name != "emotion"] + [_BrokenLayer()]
    rec = _process(db, asr)
    assert rec["status"] == "done"
    prod = json_loads(rec["producers"])
    assert "emotion" not in prod["layers"] and "prosody" in prod["layers"]
    assert "out of memory" in prod["layer_errors"]["emotion"] and "enrich" in rec["error"]
    assert len(db.transcript("u1", rec["id"])) > 0  # diarization + ASR kept

    db.models.layers = good  # "fixed": backfill clears the error
    rec = db.enrich("u1", rec["id"], ["emotion"])
    assert "emotion" in json_loads(rec["producers"])["layers"]
    assert "layer_errors" not in json_loads(rec["producers"]) and rec["error"] is None


def test_unloadable_layers_are_skipped_not_fatal(tmp_path):
    from radiodb.config import Settings
    from radiodb.service import LazyModels

    m = LazyModels(Settings(home=tmp_path, device="cpu", layers=("prosody", "bogus", "emotion"),
                            emotion_model=str(tmp_path / "no-such-model")))
    assert [l.name for l in m.layers] == ["prosody"]
    assert m.layer_errors["bogus"] == "unknown layer"
    assert "failed to load" in m.layer_errors["emotion"]
