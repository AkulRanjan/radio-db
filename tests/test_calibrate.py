import json

import numpy as np
import pytest
from typer.testing import CliRunner

from radiodb.calibrate import calibrate, der, eer, read_rttm, subtract, threshold_for_far, to_rttm, wer
from tests.conftest import FakeEmbedder, FakeTranscriber, FakeVAD, Models, conversation, script_words, wav_bytes, VOCAB

REF = [(0.0, 4.0, "alice"), (4.0, 8.0, "bob")]


# ---- metrics ---------------------------------------------------------------------------------

def test_der_perfect_and_label_permutation():
    assert der(REF, REF, collar=0)["der"] == 0
    renamed = [(0.0, 4.0, "spk7"), (4.0, 8.0, "spk2")]  # names don't matter, mapping does
    assert der(REF, renamed, collar=0)["der"] == 0


def test_der_components():
    r = der(REF, [(0.0, 8.0, "x")], collar=0)  # everything given to one speaker
    assert r["confusion"] == pytest.approx(0.5, abs=0.01) and r["miss"] == 0 and r["false_alarm"] == 0
    r = der(REF, [(0.0, 4.0, "x")], collar=0)  # bob never detected
    assert r["miss"] == pytest.approx(0.5, abs=0.01) and r["confusion"] == 0
    r = der(REF, REF + [(8.0, 10.0, "x")], collar=0)  # speech hallucinated after the end
    assert r["false_alarm"] == pytest.approx(0.25, abs=0.01)


def test_der_overlap_and_collar():
    ref = [(0.0, 4.0, "a"), (2.0, 6.0, "b")]  # 2 s overlap: 8 s of speaker time
    hyp = [(0.0, 3.0, "1"), (3.0, 6.0, "2")]  # single-speaker hypothesis can't cover overlap
    r = der(ref, hyp, collar=0)
    assert r["scored_s"] == pytest.approx(8.0, abs=0.02)
    assert r["miss"] == pytest.approx(2 / 8, abs=0.01)
    # a boundary off by 0.2 s is forgiven by a 0.25 s collar
    assert der(REF, [(0.0, 4.2, "x"), (4.2, 8.0, "y")], collar=0.25)["der"] == 0


def test_wer():
    assert wer("Hello, there! How are you?", "hello there how are you")["wer"] == 0
    r = wer("the cat sat on the mat", "the cat sat on mat today")
    assert r["errors"] == 2 and r["wer"] == pytest.approx(2 / 6)
    assert wer("Um so uh the plan is, hmm, fine", "so the plan is fine")["wer"] == 0  # fillers ignored


def test_eer_and_far_threshold():
    e, t = eer([0.9, 0.85, 0.8, 0.7], [0.1, 0.2, 0.3, 0.75])
    assert e == pytest.approx(0.25) and 0.7 <= t <= 0.75
    assert eer([0.9, 0.8], [0.1, 0.2]) == (0, pytest.approx(0.5))  # separable: middle of the gap
    t = threshold_for_far([0.1, 0.2, 0.3, 0.75], 0.0)
    assert t > 0.75  # zero false accepts allowed


def test_subtract_keeps_solo_parts():
    assert subtract((0.0, 10.0), [(2.0, 3.0), (5.0, 12.0)]) == [(0.0, 2.0), (3.0, 5.0)]
    assert subtract((0.0, 1.0), [(0.0, 1.0)]) == []
    assert subtract((0.0, 1.0), [(2.0, 3.0)]) == [(0.0, 1.0)]


def test_rttm_round_trip(tmp_path):
    p = tmp_path / "x.rttm"
    p.write_text(to_rttm("x", REF))
    assert read_rttm(p) == REF


# ---- end to end on a synthetic labelled dataset ----------------------------------------------

def _dataset(tmp_path):
    """Three 'conversations' with recurring speakers A (alice) and B (bob), plus C (carol)."""
    scripts = {
        "one": [("A", 4.0), (None, 0.6), ("B", 4.0), (None, 0.6), ("A", 3.0)],
        "two": [("B", 3.0), (None, 0.6), ("C", 4.0)],
        "three": [("A", 3.0), (None, 0.6), ("C", 3.0), (None, 0.6), ("B", 3.0)],
    }
    names = {"A": "alice", "B": "bob", "C": "carol"}
    for stem, script in scripts.items():
        (tmp_path / f"{stem}.wav").write_bytes(wav_bytes(conversation(script)))
        t, rows = 0.0, []
        for v, d in script:
            if v:
                rows.append((t, t + d, names[v]))
            t += d
        (tmp_path / f"{stem}.rttm").write_text(to_rttm(stem, rows))
    (tmp_path / "one.txt").write_text(" ".join(w.w for w in script_words(scripts["one"], VOCAB)))
    return scripts


def test_calibrate_recovers_good_settings(tmp_path):
    scripts = _dataset(tmp_path)
    asr = FakeTranscriber()
    asr.next_words = script_words(scripts["one"], VOCAB)
    models = Models(FakeVAD(), FakeEmbedder(), asr)
    rep = calibrate(tmp_path, models, thresholds=(0.05, 0.3, 0.6, 0.9, 1.5), min_speaker_s=(1.0, 5.0),
                    progress=lambda *_: None)

    best = rep["diarization"]["best"]
    plateau = rep["diarization"]["plateau"]
    assert best["der"] < 0.02
    assert len(plateau) >= 3 and (best["diarization_threshold"], best["min_speaker_s"]) in plateau
    assert best["diarization_threshold"] not in (0.05, 0.9, 1.5)  # middle of the plateau, not an edge
    worst = rep["diarization"]["grid"][-1]
    assert worst["der"] > 0.3 and worst["diarization_threshold"] == 1.5  # merges everyone into one speaker

    ident = rep["identification"]
    assert ident["genuine_pairs"] == 5  # same name in different files: alice 1, bob 3, carol 1
    assert ident["eer"] == 0
    assert max(ident["impostor_score_range"]) < ident["recommended_threshold"] <= min(ident["genuine_score_range"])
    assert rep["asr"]["wer"] == 0 and rep["files"][0]["wer"] == 0
    assert rep["recommended_env"]["RADIODB_DIARIZATION_THRESHOLD"] == best["diarization_threshold"]


def test_calibrate_cli(tmp_path, monkeypatch):
    from radiodb import cli
    from radiodb.service import RadioDB

    data = tmp_path / "set"
    data.mkdir()
    _dataset(data)
    models = Models(FakeVAD(), FakeEmbedder(), FakeTranscriber())
    from radiodb.config import Settings

    monkeypatch.setattr(cli, "_db", lambda: RadioDB(Settings(home=tmp_path / "home"), models=models))
    out = tmp_path / "report.json"
    res = CliRunner().invoke(cli.app, ["calibrate", str(data), "--out", str(out), "--no-asr"])
    assert res.exit_code == 0, res.output
    assert "recommended settings" in res.output and "RADIODB_SPEAKER_MATCH_THRESHOLD" in res.output
    assert json.loads(out.read_text())["identification"]["eer"] == 0


def test_calibrate_needs_labels(tmp_path):
    (tmp_path / "x.wav").write_bytes(wav_bytes(conversation([("A", 2.0)])))
    with pytest.raises(ValueError, match="no audio files"):
        calibrate(tmp_path, Models(FakeVAD(), FakeEmbedder(), FakeTranscriber()))
