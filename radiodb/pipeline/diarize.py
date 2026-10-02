"""Diarization (who spoke when) and alignment of transcript words to speakers.

Approach (the same shape as pyannote / WhisperX, built from non-gated models):
  1. VAD finds speech regions.
  2. Each region is cut into short overlapping windows and embedded with a speaker encoder.
  3. Windows are clustered by cosine distance; one cluster = one voice in this recording.
     Short windows make ECAPA noisy, so clustering deliberately over-splits; clusters with less
     than ``min_speaker_s`` of speech are then folded into their nearest large cluster.
  4. Consecutive windows with the same label are merged into turns.
  5. ASR words are assigned to the turn they overlap most; words are grouped into segments.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..audio import MODEL_SR
from .interfaces import VAD, SpeakerEmbedder, Span, Turn, Word

WINDOW_S = 1.5
HOP_S = 0.75
MIN_WINDOW_S = 0.5


def windows(spans: list[Span]) -> list[Span]:
    out: list[Span] = []
    for sp in spans:
        if sp.end - sp.start < MIN_WINDOW_S:
            continue
        t = sp.start
        while True:
            end = min(t + WINDOW_S, sp.end)
            if end - t >= MIN_WINDOW_S:
                out.append(Span(t, end))
            if end >= sp.end:
                break
            t += HOP_S
    return out


def cluster(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    if len(embeddings) == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold
    )
    return model.fit_predict(embeddings)


def turns_from_windows(wins: list[Span], labels: np.ndarray) -> list[Turn]:
    """Each window owns the region from its start to the next window's start (or its end)."""
    turns: list[Turn] = []
    for i, (w, lab) in enumerate(zip(wins, labels)):
        start = w.start if not turns or turns[-1].end < w.start else turns[-1].end
        end = w.end
        if i + 1 < len(wins) and wins[i + 1].start < w.end:
            end = wins[i + 1].start
        if end <= start:
            continue
        if turns and turns[-1].cluster == lab and start - turns[-1].end < 0.5:
            turns[-1].end = end
        else:
            turns.append(Turn(start, end, int(lab)))
    return turns


def absorb_small_clusters(
    emb: np.ndarray, labels: np.ndarray, wins: list[Span], min_speaker_s: float
) -> np.ndarray:
    """Merge clusters with under ``min_speaker_s`` of speech into the most similar large one."""
    labels = labels.copy()
    while True:
        ids = np.unique(labels)
        dur = {c: sum(t.end - t.start for t in turns_from_windows(wins, labels) if t.cluster == c) for c in ids}
        big = [c for c in ids if dur[c] >= min_speaker_s]
        small = [c for c in ids if dur[c] < min_speaker_s]
        if not small or not big:
            return labels
        cent = {c: _normalise(emb[labels == c].mean(axis=0)) for c in ids}
        c = min(small, key=lambda c: dur[c])
        labels[labels == c] = max(big, key=lambda b: float(cent[b] @ cent[c]))


def diarize(
    audio16k: np.ndarray, vad: VAD, embedder: SpeakerEmbedder, threshold: float,
    min_speaker_s: float = 3.0,
) -> tuple[list[Turn], dict[int, np.ndarray]]:
    """Return speaker turns and one L2-normalised centroid per cluster."""
    wins = windows(vad.speech(audio16k))
    if not wins:
        return [], {}
    clips = [audio16k[int(w.start * MODEL_SR): int(w.end * MODEL_SR)] for w in wins]
    emb = embedder.embed(clips)
    labels = absorb_small_clusters(emb, cluster(emb, threshold), wins, min_speaker_s)
    centroids = {int(c): _normalise(emb[labels == c].mean(axis=0)) for c in np.unique(labels)}
    return turns_from_windows(wins, labels), centroids


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_words(words: list[Word], turns: list[Turn]) -> list[int | None]:
    """Cluster label per word: the turn with most overlap, else the nearest turn."""
    out: list[int | None] = []
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = _overlap(w.start, w.end, t.start, t.end)
            if ov > best_ov:
                best, best_ov = t.cluster, ov
        if best is None and turns:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).cluster
        out.append(best)
    return out


def build_segments(
    words: list[Word], labels: list[int | None], turns: list[Turn], max_gap_s: float = 1.0
) -> list[dict]:
    """Group words into segments: split on speaker change or a pause longer than max_gap_s.

    Turns with no words (speech ASR missed, laughter, etc.) become text-less segments so that
    no voiced audio is dropped from the index.
    """
    segs: list[dict] = []
    for w, lab in zip(words, labels):
        cur = segs[-1] if segs else None
        if cur and cur["cluster"] == lab and w.start - cur["end_s"] <= max_gap_s:
            cur["end_s"] = max(cur["end_s"], w.end)
            cur["words"].append(w)
        else:
            segs.append({"cluster": lab, "start_s": w.start, "end_s": w.end, "words": [w]})

    for t in turns:
        has_words = any(t.start <= (w.start + w.end) / 2 <= t.end for w in words)
        if t.end - t.start >= 1.0 and not has_words:
            segs.append({"cluster": t.cluster, "start_s": t.start, "end_s": t.end, "words": []})

    segs.sort(key=lambda s: s["start_s"])
    for s in segs:
        s["text"] = " ".join(w.w.strip() for w in s["words"]).strip()
        s["words"] = [
            {"w": w.w.strip(), "start": round(w.start, 3), "end": round(w.end, 3), "p": round(w.p, 3)}
            for w in s["words"]
        ]
    return segs


def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 0 else v.astype(np.float32)
