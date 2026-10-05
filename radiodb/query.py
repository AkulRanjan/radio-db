"""Query helpers: speaker-relative speaking style and rank fusion.

Raw prosody numbers are hard to search ("what is a high pitch?"), and they vary a lot
between people. So each segment is described relative to *that speaker's own* baseline:
a segment is "fast" if it is fast for this person, not compared with everyone else.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

RRF_K = 60
MIN_BASELINE = 5   # segments needed before a speaker's baseline is trusted
Z_THRESHOLD = 1.0  # |robust z| at or above this is labelled; below is "typical"

# style dimension -> (prosody feature, baseline scope, (low label, high label))
#  - pitch, pace and pitch range are habits of the speaker: one baseline across all recordings
#  - loudness depends on mic and gain: baseline per speaker *within one recording*
STYLE_DIMENSIONS = {
    "pitch": ("f0_mean_st", "speaker", ("low", "high")),
    "pace": ("articulation_rate_sps", "speaker", ("slow", "fast")),
    "volume": ("intensity_db", "speaker_recording", ("quiet", "loud")),
    "expressiveness": ("f0_range_st", "speaker", ("flat", "animated")),
}
STYLE_VALUES = {dim: {*labels, "typical"} for dim, (_, _, labels) in STYLE_DIMENSIONS.items()}


def _robust_z(values: dict[int, float]) -> dict[int, float]:
    if len(values) < MIN_BASELINE:
        return {}
    arr = np.array(list(values.values()))
    med = float(np.median(arr))
    scale = 1.4826 * float(np.median(np.abs(arr - med)))  # MAD scaled to match std for normal data
    scale = max(scale, 0.1 * float(np.std(arr)), 1e-6)
    return {k: (v - med) / scale for k, v in values.items()}


def style_table(segments: list[dict], prosody: dict[int, dict]) -> dict[int, dict]:
    """{segment_id: {dim: label, ..., "z": {dim: z}}} for segments with a speaker and prosody."""
    out: dict[int, dict] = defaultdict(lambda: {"z": {}})
    for dim, (feat, scope, (lo, hi)) in STYLE_DIMENSIONS.items():
        groups: dict[tuple, dict[int, float]] = defaultdict(dict)
        for s in segments:
            p = prosody.get(s["id"])
            if not p or s.get("speaker_id") is None or p.get(feat) is None:
                continue
            key = (s["speaker_id"],) if scope == "speaker" else (s["speaker_id"], s["recording_id"])
            groups[key][s["id"]] = float(p[feat])
        for vals in groups.values():
            for seg_id, z in _robust_z(vals).items():
                out[seg_id][dim] = hi if z >= Z_THRESHOLD else lo if z <= -Z_THRESHOLD else "typical"
                out[seg_id]["z"][dim] = round(z, 2)
    return dict(out)


def rrf(*rankings: list[int], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal-rank fusion of ranked id lists (best first)."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] += 1.0 / (k + rank + 1)
    return dict(scores)
