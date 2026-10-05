"""Calibrate radio-db on your own labelled conversations.

Dataset: a directory of audio files, each with a same-named ``.rttm`` (who spoke when) and
optionally a ``.txt`` (reference transcript). Speaker names in the RTTMs must be consistent
across files ("alice" is the same person everywhere); that is what calibrates speaker
identification. A good set is 20–30 minutes from real users, several files, with recurring speakers.

Tip: ``radiodb rttm <recording>`` exports radio-db's own diarization as RTTM. Correcting that
in a text editor or labelling tool is much faster than labelling from scratch.

Reports (and recommends settings for):
  * diarization DER, sweeping RADIODB_DIARIZATION_THRESHOLD × RADIODB_MIN_SPEAKER_S
  * speaker identification EER + low-false-accept threshold → RADIODB_SPEAKER_MATCH_THRESHOLD
  * ASR word error rate (when .txt references exist)
  * real-time factor per stage on this hardware
"""

from __future__ import annotations

import itertools
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import audio
from .pipeline.diarize import cluster_turns, embed_windows
from .pipeline.interfaces import Turn

PLATEAU_TOL = 0.005  # settings within 0.5 % DER of the best count as equally good
AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".aiff", ".aif"}

# ---- formats --------------------------------------------------------------------------------


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """RTTM ``SPEAKER <file> <chan> <onset> <dur> <NA> <NA> <name> ...`` → [(start, end, name)]."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 8 and parts[0] == "SPEAKER":
            start, dur = float(parts[3]), float(parts[4])
            out.append((start, start + dur, parts[7]))
    return sorted(out)


def to_rttm(file_id: str, segments: list[tuple[float, float, str]]) -> str:
    return "".join(
        f"SPEAKER {file_id} 1 {s:.3f} {e - s:.3f} <NA> <NA> {name} <NA> <NA>\n" for s, e, name in segments if e > s
    )


# ---- metrics --------------------------------------------------------------------------------


def der(
    ref: list[tuple[float, float, str]], hyp: list[tuple[float, float, str]],
    collar: float = 0.25, step: float = 0.01,
) -> dict:
    """Diarization error rate, NIST-style, on a frame grid.

    Per frame: error = max(n_ref, n_hyp) - n_correct, where n_correct uses the optimal one-to-one
    mapping of hypothesis to reference speakers (Hungarian). Frames within ``collar`` seconds of
    a reference boundary are not scored. Overlapping reference speech counts once per speaker.
    """
    end = max([e for _, e, _ in ref + hyp] + [0.0])
    n = int(np.ceil(end / step)) + 1
    ref_names = sorted({r[2] for r in ref})
    hyp_names = sorted({h[2] for h in hyp})
    R = np.zeros((len(ref_names), n), dtype=bool)
    H = np.zeros((len(hyp_names), n), dtype=bool)
    for s, e, name in ref:
        R[ref_names.index(name), int(round(s / step)): int(round(e / step))] = True
    for s, e, name in hyp:
        H[hyp_names.index(name), int(round(s / step)): int(round(e / step))] = True

    scored = np.ones(n, dtype=bool)
    if collar > 0:
        c = int(round(collar / step))
        for s, e, _ in ref:
            for b in (int(round(s / step)), int(round(e / step))):
                scored[max(0, b - c): b + c] = False
    R, H = R[:, scored], H[:, scored]

    # optimal speaker mapping maximises total co-occurring time
    overlap = (R[:, None, :] & H[None, :, :]).sum(axis=2) if len(ref_names) and len(hyp_names) else np.zeros((0, 0))
    mapping = {}
    if overlap.size:
        ri, hi = linear_sum_assignment(-overlap)
        mapping = {hyp_names[h]: ref_names[r] for r, h in zip(ri, hi) if overlap[r, h] > 0}

    n_ref, n_hyp = R.sum(axis=0), H.sum(axis=0)
    correct = np.zeros(R.shape[1], dtype=int)
    for h_name, r_name in mapping.items():
        correct += R[ref_names.index(r_name)] & H[hyp_names.index(h_name)]
    total = int(n_ref.sum())
    miss = int(np.clip(n_ref - n_hyp, 0, None).sum())
    fa = int(np.clip(n_hyp - n_ref, 0, None).sum())
    conf = int((np.minimum(n_ref, n_hyp) - correct).sum())
    return {
        "der": (miss + fa + conf) / total if total else 0.0,
        "miss": miss / total if total else 0.0,
        "false_alarm": fa / total if total else 0.0,
        "confusion": conf / total if total else 0.0,
        "scored_s": round(total * step, 2),
        "ref_speakers": len(ref_names),
        "hyp_speakers": len(hyp_names),
    }


# Hesitations / backchannels that transcription conventions disagree on; removed from both sides
# before scoring, as is standard for conversational-speech WER.
FILLERS = {"um", "uh", "er", "erm", "ah", "eh", "hm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "huh"}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text.lower()) if w not in FILLERS]


def wer(ref_text: str, hyp_text: str) -> dict:
    """Word error rate after lowercasing and stripping punctuation (Levenshtein over words)."""
    r, h = _words(ref_text), _words(hyp_text)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return {"wer": d[-1, -1] / len(r) if r else 0.0, "errors": int(d[-1, -1]), "ref_words": len(r)}


def eer(genuine: list[float], impostor: list[float]) -> tuple[float, float]:
    """Equal error rate and its threshold: where false-accept ≈ false-reject rate.
    Accept means score >= threshold."""
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    if i.max() < g.min():  # perfectly separable: any threshold in the gap works; take its middle
        return 0.0, float((i.max() + g.min()) / 2)
    best_gap, best = np.inf, (1.0, 0.0)
    for t in np.unique(np.concatenate([g, i])):
        far, frr = float((i >= t).mean()), float((g < t).mean())
        if abs(far - frr) < best_gap:
            best_gap, best = abs(far - frr), ((far + frr) / 2, float(t))
    return best


def threshold_for_far(impostor: list[float], far: float) -> float:
    """Lowest threshold (accept = score >= t) at which at most ``far`` of impostors are accepted."""
    i = np.sort(np.asarray(impostor, dtype=float))
    if not len(i):
        return 0.0
    allowed = int(np.floor(far * len(i)))  # impostors we may accept
    if allowed >= len(i):
        return float(i[0])
    return float(i[len(i) - allowed - 1] + 1e-6)  # just above the highest impostor we must reject


# ---- calibration run ------------------------------------------------------------------------


@dataclass
class Item:
    audio: Path
    rttm: Path
    txt: Path | None


def find_items(directory: Path) -> list[Item]:
    items = []
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in AUDIO_EXT and p.with_suffix(".rttm").exists():
            txt = p.with_suffix(".txt")
            items.append(Item(p, p.with_suffix(".rttm"), txt if txt.exists() else None))
    return items


def subtract(span: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Parts of ``span`` not covered by any of ``cuts``."""
    parts = [span]
    for c0, c1 in cuts:
        nxt = []
        for p0, p1 in parts:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        parts = nxt
    return parts


def _turns_to_segments(turns: list[Turn]) -> list[tuple[float, float, str]]:
    return [(t.start, t.end, f"spk{t.cluster}") for t in turns]


def calibrate(
    directory: Path, models, *, thresholds=(0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8),
    min_speaker_s=(1.0, 2.0, 3.0, 5.0), collar: float = 0.25, target_far: float = 0.01,
    run_asr: bool = True, progress=print,
) -> dict:
    items = find_items(directory)
    if not items:
        raise ValueError(f"no audio files with matching .rttm found in {directory}")

    timing = {"audio_s": 0.0, "diarize_embed_s": 0.0, "asr_s": 0.0}
    per_file, sweep = [], {(t, m): [] for t, m in itertools.product(thresholds, min_speaker_s)}
    speaker_vecs: list[tuple[str, str, np.ndarray]] = []  # (file, ref speaker, voiceprint)
    asr_totals = {"errors": 0, "ref_words": 0}

    for it in items:
        progress(f"  {it.audio.name}")
        a16 = audio.to_model_input(audio.decode(it.audio.read_bytes()))
        ref = read_rttm(it.rttm)
        timing["audio_s"] += len(a16) / audio.MODEL_SR

        t0 = time.perf_counter()
        wins, emb = embed_windows(a16, models.vad, models.embedder)
        timing["diarize_embed_s"] += time.perf_counter() - t0
        for (t, m) in sweep:
            turns, _ = cluster_turns(wins, emb, t, m)
            sweep[(t, m)].append((der(ref, _turns_to_segments(turns), collar), len(ref)))

        # one voiceprint per reference speaker per file, from the parts of their speech where
        # nobody else is talking
        for name in sorted({r[2] for r in ref}):
            others = [(s2, e2) for s2, e2, n2 in ref if n2 != name]
            solo = [part for s, e, n in ref if n == name for part in subtract((s, e), others)]
            # 10 s pieces, as in enrollment (`RadioDB._voiceprint`)
            clips = [a16[int(t * audio.MODEL_SR): int(min(t + 10.0, e) * audio.MODEL_SR)]
                     for s, e in solo if e - s >= 1.0 for t in np.arange(s, e, 10.0) if min(t + 10.0, e) - t >= 1.0]
            if clips:
                v = models.embedder.embed(clips)
                w = np.array([len(c) for c in clips], dtype=np.float64)[:, None]
                c = (v * w).sum(axis=0)
                speaker_vecs.append((it.audio.name, name, (c / np.linalg.norm(c)).astype(np.float32)))

        file_report = {"file": it.audio.name, "duration_s": round(len(a16) / audio.MODEL_SR, 1),
                       "ref_speakers": len({r[2] for r in ref})}
        if run_asr and it.txt:
            t0 = time.perf_counter()
            hyp = " ".join(w.w for w in models.transcriber.transcribe(a16).words)
            timing["asr_s"] += time.perf_counter() - t0
            w_ = wer(it.txt.read_text(encoding="utf-8"), hyp)
            asr_totals["errors"] += w_["errors"]
            asr_totals["ref_words"] += w_["ref_words"]
            file_report["wer"] = round(w_["wer"], 4)
        per_file.append(file_report)

    # aggregate DER per setting (weighted by scored time)
    grid = []
    for (t, m), results in sweep.items():
        scored = sum(r["scored_s"] for r, _ in results)
        agg = {k: sum(r[k] * r["scored_s"] for r, _ in results) / scored if scored else 0.0
               for k in ("der", "miss", "false_alarm", "confusion")}
        spk_err = float(np.mean([abs(r["hyp_speakers"] - r["ref_speakers"]) for r, _ in results]))
        grid.append({"diarization_threshold": t, "min_speaker_s": m, **{k: round(v, 4) for k, v in agg.items()},
                     "mean_speaker_count_error": round(spk_err, 2)})
    grid.sort(key=lambda g: (g["der"], g["mean_speaker_count_error"]))
    # Several settings often tie. Pick the middle of the near-best plateau, not its edge: it is
    # the choice least likely to tip over on new recordings.
    plateau = [g for g in grid if g["der"] <= grid[0]["der"] + PLATEAU_TOL]
    mid_t = float(np.median([g["diarization_threshold"] for g in plateau]))
    mid_m = float(np.median([g["min_speaker_s"] for g in plateau]))
    best = min(plateau, key=lambda g: (abs(g["diarization_threshold"] - mid_t), abs(g["min_speaker_s"] - mid_m),
                                       g["der"], g["diarization_threshold"]))
    for f, (r, _) in zip(per_file, sweep[(best["diarization_threshold"], best["min_speaker_s"])]):
        f["der_at_best"] = round(r["der"], 4)

    # speaker identification: same-name pairs across different files are genuine
    genuine, impostor = [], []
    for (f1, n1, v1), (f2, n2, v2) in itertools.combinations(speaker_vecs, 2):
        if f1 == f2:
            if n1 != n2:
                impostor.append(float(v1 @ v2))
        elif n1 == n2:
            genuine.append(float(v1 @ v2))
        else:
            impostor.append(float(v1 @ v2))
    ident: dict = {"genuine_pairs": len(genuine), "impostor_pairs": len(impostor)}
    if genuine and impostor:
        e, t_eer = eer(genuine, impostor)
        t_far = threshold_for_far(impostor, target_far)
        ident.update({
            "eer": round(e, 4), "threshold_at_eer": round(t_eer, 3),
            f"threshold_at_far_{target_far:g}": round(t_far, 3),
            "frr_at_that_threshold": round(float((np.asarray(genuine) < t_far).mean()), 4),
            "genuine_score_range": [round(min(genuine), 3), round(max(genuine), 3)],
            "impostor_score_range": [round(min(impostor), 3), round(max(impostor), 3)],
            # Prefer few false merges (two people becoming one identity is hard to undo; splits
            # are fixed with `speakers merge`), but never go below the EER point.
            "recommended_threshold": round(max(t_eer, t_far), 3),
        })
    else:
        ident["note"] = ("need the same speaker name in at least two files (genuine pairs) and "
                         "different speakers (impostor pairs) to calibrate identification")

    audio_s = timing["audio_s"]
    report = {
        "files": per_file,
        "total_audio_s": round(audio_s, 1),
        "diarization": {"best": best, "collar_s": collar,
                        "plateau": [(g["diarization_threshold"], g["min_speaker_s"]) for g in plateau],
                        "grid": grid},
        "identification": ident,
        "asr": ({"wer": round(asr_totals["errors"] / asr_totals["ref_words"], 4), **asr_totals}
                if asr_totals["ref_words"] else None),
        "real_time_factor": {
            "diarization_embedding": round(timing["diarize_embed_s"] / audio_s, 3) if audio_s else None,
            "asr": round(timing["asr_s"] / audio_s, 3) if audio_s and timing["asr_s"] else None,
        },
    }
    report["recommended_env"] = {
        "RADIODB_DIARIZATION_THRESHOLD": best["diarization_threshold"],
        "RADIODB_MIN_SPEAKER_S": best["min_speaker_s"],
        **({"RADIODB_SPEAKER_MATCH_THRESHOLD": ident["recommended_threshold"]}
           if "recommended_threshold" in ident else {}),
    }
    return report
