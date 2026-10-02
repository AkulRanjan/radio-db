"""Fakes: each 'voice' is a pure tone, the embedder maps pitch to a direction, and the
transcriber returns scripted words. That exercises the real pipeline logic without model weights."""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from radiodb.catalog import Catalog
from radiodb.blobstore import LocalBlobStore
from radiodb.config import Settings
from radiodb.pipeline.interfaces import Span, Transcript, Word
from radiodb.service import Models, RadioDB

SR = 16_000
VOICES = {"A": 220.0, "B": 700.0, "C": 1500.0}


def tone(freq: float, seconds: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def conversation(script: list[tuple[str | None, float]], sr: int = SR) -> np.ndarray:
    """script: [(voice or None for silence, seconds), ...]"""
    return np.concatenate([tone(VOICES[v], d, sr) if v else silence(d, sr) for v, d in script])


def wav_bytes(samples: np.ndarray, sr: int = SR, fmt: str = "WAV", subtype: str | None = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def script_words(script: list[tuple[str | None, float]], vocab: dict[str, list[str]]) -> list[Word]:
    """One scripted word per second of each voiced span."""
    words, t = [], 0.0
    for v, d in script:
        if v:
            n = max(1, int(d))
            for i in range(n):
                words.append(Word(vocab[v][i % len(vocab[v])], t + i * d / n, t + (i + 0.8) * d / n))
        t += d
    return words


class FakeVAD:
    name = "fake-vad"

    def speech(self, a: np.ndarray) -> list[Span]:
        frame = SR // 50
        n = len(a) // frame
        voiced = np.sqrt((a[: n * frame].reshape(n, frame) ** 2).mean(axis=1)) > 0.01
        spans, start = [], None
        for i, v in enumerate(voiced):
            if v and start is None:
                start = i
            if not v and start is not None:
                spans.append(Span(start * frame / SR, i * frame / SR)); start = None
        if start is not None:
            spans.append(Span(start * frame / SR, n * frame / SR))
        return spans


class FakeEmbedder:
    name = "fake-embedder"
    dim = 8

    def embed(self, clips):
        out = []
        for c in clips:
            spec = np.abs(np.fft.rfft(c))
            f = np.argmax(spec) * SR / len(c)
            v = np.full(self.dim, 0.02, dtype=np.float32)
            v[min(int(f // 200), self.dim - 1)] = 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class FakeTranscriber:
    name = "fake-asr"

    def __init__(self):
        self.next_words: list[Word] = []

    def transcribe(self, a):
        return Transcript(words=list(self.next_words), model=self.name)


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path, speaker_match_threshold=0.8, diarization_threshold=0.3, min_speaker_s=1.0)


@pytest.fixture
def asr():
    return FakeTranscriber()


@pytest.fixture
def db(settings, asr):
    d = RadioDB(
        settings,
        catalog=Catalog(settings.db_path),
        blobs=LocalBlobStore(settings.blob_dir),
        models=Models(vad=FakeVAD(), embedder=FakeEmbedder(), transcriber=asr),
    )
    yield d
    d.catalog.close()


VOCAB = {
    "A": ["hello", "how", "is", "the", "budget", "looking"],
    "B": ["pretty", "good", "we", "saved", "money", "today"],
    "C": ["sorry", "i", "am", "late", "again", "folks"],
}
