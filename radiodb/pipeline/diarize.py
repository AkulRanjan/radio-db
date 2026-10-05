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


MIN_SPEAKER_FRACTION = 0.1


def effective_min_speaker_s(wins: list[Span], min_speaker_s: float) -> float:
    """The speech a voice needs to count as its own speaker: ``min_speaker_s`` (calibrated on
    long meetings), but never more than 10 % of the recording's speech, so short calls keep
    speakers who only talk for a few seconds."""
    return min(min_speaker_s, MIN_SPEAKER_FRACTION * float(window_ownership(wins).sum()))


def window_ownership(wins: list[Span]) -> np.ndarray:
    """Seconds of audio each window owns: the same regions ``turns_from_windows`` assigns
    (start to the next window's start), independent of labels."""
    own = np.zeros(len(wins))
    prev_end = -np.inf
    for i, w in enumerate(wins):
        start = max(w.start, prev_end)
        end = wins[i + 1].start if i + 1 < len(wins) and wins[i + 1].start < w.end else w.end
        if end > start:
            own[i] = end - start
            prev_end = end
    return own


def absorb_small_clusters(
    emb: np.ndarray, labels: np.ndarray, wins: list[Span], min_speaker_s: float
) -> np.ndarray:
    """Merge clusters with under ``min_speaker_s`` of speech into the most similar large one,
    smallest first. Durations and centroid sums are updated incrementally, so this stays fast
    for hour-long recordings with hundreds of initial clusters."""
    ids, inv = np.unique(labels, return_inverse=True)
    k = len(ids)
    dur = np.bincount(inv, weights=window_ownership(wins), minlength=k)
    sums = np.zeros((k, emb.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, emb)
    target = np.arange(k)
    alive = np.ones(k, dtype=bool)
    while True:
        small = np.where(alive & (dur < min_speaker_s))[0]
        big = np.where(alive & (dur >= min_speaker_s))[0]
        if not len(small) or not len(big):
            break
        c = small[np.argmin(dur[small])]
        cent = sums / np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
        t = big[np.argmax(cent[big] @ cent[c])]
        sums[t] += sums[c]
        dur[t] += dur[c]
        alive[c] = False
        target[c] = t  # targets are always big clusters, which are never merged away
    return ids[target[inv]]


def diarize(
    audio16k: np.ndarray, vad: VAD, embedder: SpeakerEmbedder, threshold: float,
    min_speaker_s: float = 3.0,
) -> tuple[list[Turn], dict[int, np.ndarray]]:
    """Return speaker turns and one L2-normalised centroid per cluster."""
    wins, emb = embed_windows(audio16k, vad, embedder)
    return cluster_turns(wins, emb, threshold, min_speaker_s)


def embed_windows(audio16k: np.ndarray, vad: VAD, embedder: SpeakerEmbedder) -> tuple[list[Span], np.ndarray]:
    """VAD + sliding windows + speaker embeddings: the expensive half of diarization."""
    wins = windows(vad.speech(audio16k))
    if not wins:
        return [], np.zeros((0, embedder.dim), dtype=np.float32)
    clips = [audio16k[int(w.start * MODEL_SR): int(w.end * MODEL_SR)] for w in wins]
    return wins, embedder.embed(clips)


def cluster_turns(
    wins: list[Span], emb: np.ndarray, threshold: float, min_speaker_s: float
) -> tuple[list[Turn], dict[int, np.ndarray]]:
    """The cheap half: cluster window embeddings into speakers and turns (re-runnable per setting)."""
    if not wins:
        return [], {}
    labels = absorb_small_clusters(emb, cluster(emb, threshold), wins, effective_min_speaker_s(wins, min_speaker_s))
    centroids = {int(c): _normalise(emb[labels == c].mean(axis=0)) for c in np.unique(labels)}
    return turns_from_windows(wins, labels), centroids


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_words(words: list[Word], turns: list[Turn]) -> list[int | None]:
    """Cluster label per word: the turn with most overlap. A word overlapping no turn (often a
    zero-length ASR timestamp at a speaker change) stays with the previous word's speaker if it
    follows closely, else goes to the nearest turn."""
    out: list[int | None] = []
    prev_end = None
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            ov = _overlap(w.start, w.end, t.start, t.end)
            if ov > best_ov:
                best, best_ov = t.cluster, ov
        if best is None and out and out[-1] is not None and prev_end is not None and w.start - prev_end <= 0.3:
            best = out[-1]
        if best is None and turns:
            mid = (w.start + w.end) / 2
            best = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).cluster
        out.append(best)
        prev_end = w.end
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
