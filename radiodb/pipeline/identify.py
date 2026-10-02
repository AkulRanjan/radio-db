"""Speaker identification: map per-recording voice clusters to a user's persistent speakers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    speaker_id: str | None
    score: float


def best_match(
    centroid: np.ndarray, candidates: list[tuple[str, np.ndarray]], threshold: float
) -> Match:
    if not candidates:
        return Match(None, 0.0)
    ids = [c[0] for c in candidates]
    sims = np.stack([c[1] for c in candidates]) @ centroid
    i = int(np.argmax(sims))
    score = float(sims[i])
    return Match(ids[i] if score >= threshold else None, score)


def running_mean(centroid: np.ndarray, n: int, new: np.ndarray, new_n: int = 1) -> np.ndarray:
    v = centroid * n + new * new_n
    return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)
