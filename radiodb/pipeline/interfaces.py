"""Model interfaces. Real backends live in ``backends.py``; tests use lightweight fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Span:
    start: float
    end: float


@dataclass
class Word:
    w: str
    start: float
    end: float
    p: float = 1.0


@dataclass
class Turn:
    """A stretch of audio attributed to one diarization cluster."""

    start: float
    end: float
    cluster: int


@dataclass
class Transcript:
    words: list[Word] = field(default_factory=list)
    model: str = ""


class VAD(Protocol):
    name: str

    def speech(self, audio16k: np.ndarray) -> list[Span]: ...


class SpeakerEmbedder(Protocol):
    name: str
    dim: int

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings, shape (len(clips), dim). Clips are mono 16 kHz float32."""
        ...


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio16k: np.ndarray) -> Transcript: ...
