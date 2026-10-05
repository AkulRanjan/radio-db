"""Fakes: each 'voice' is a pure tone, the embedder maps pitch to a direction, and the
transcriber returns scripted words. That exercises the real pipeline logic without model weights."""

from __future__ import annotations

import io
import os
import uuid

import numpy as np
import pytest
import soundfile as sf

from radiodb.catalog import SQLiteCatalog
from radiodb.blobstore import LocalBlobStore
from radiodb.config import Settings
from radiodb.pipeline.interfaces import Span, Transcript, Word
from radiodb.service import Models, RadioDB

SR = 16_000
VOICES = {"A": 220.0, "B": 700.0, "C": 1500.0}


def tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def conversation(script: list[tuple], sr: int = SR) -> np.ndarray:
    """script: [(voice or None for silence, seconds[, amplitude]), ...]"""
    return np.concatenate([
        tone(VOICES[v], d, sr, *rest) if v else silence(d, sr) for v, d, *rest in script
    ])


def wav_bytes(samples: np.ndarray, sr: int = SR, fmt: str = "WAV", subtype: str | None = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def script_words(script: list[tuple[str | None, float]], vocab: dict[str, list[str]]) -> list[Word]:
    """One scripted word per second of each voiced span."""
    words, t = [], 0.0
    for v, d, *_ in script:
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


CONCEPTS = {
    "finance": {"budget", "money", "saved", "cost", "finances", "spending", "expensive"},
    "greeting": {"hello", "hi", "morning", "greetings"},
    "lateness": {"late", "sorry", "delay", "tardy"},
}


class FakeTextEmbedder:
    """Maps words to hand-made concepts so 'finances' matches 'budget' without sharing a word."""

    name = "fake-text"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.full(len(CONCEPTS) + 1, 0.01, dtype=np.float32)
            for w in t.lower().replace(".", " ").split():
                for i, words in enumerate(CONCEPTS.values()):
                    if w in words:
                        v[i] += 1
            if v[:-1].max() < 1:  # no known concept: point at the "other" axis, not everywhere
                v[-1] = 1
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class FakeEmotionClassifier:
    """Loud -> angry, very quiet -> sad, otherwise neutral (by RMS of the clip)."""

    name = "fake-emotion"
    labels = ["neutral", "happy", "angry", "sad"]

    def probs(self, clip):
        rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
        p = {"angry": [0.1, 0.05, 0.8, 0.05], "sad": [0.2, 0.05, 0.05, 0.7]}.get(
            "angry" if rms > 0.4 else "sad" if rms < 0.05 else "", [0.6, 0.2, 0.1, 0.1])
        return np.array(p)


def default_layers():
    from radiodb.pipeline.layers import EmotionLayer, SemanticLayer

    layers = [SemanticLayer(FakeTextEmbedder()), EmotionLayer(FakeEmotionClassifier())]
    try:
        from radiodb.pipeline.layers import ProsodyLayer

        layers.insert(0, ProsodyLayer())
    except ImportError:  # parselmouth not installed
        pass
    return layers


@pytest.fixture
def settings(tmp_path):
    return Settings(home=tmp_path, speaker_match_threshold=0.8, diarization_threshold=0.3, min_speaker_s=1.0)


@pytest.fixture
def asr():
    return FakeTranscriber()


PG_URL = os.environ.get("RADIODB_TEST_DATABASE_URL")


@pytest.fixture(params=["sqlite", "postgres"])
def catalog(request, settings):
    """Every catalog-backed test runs on SQLite and, when RADIODB_TEST_DATABASE_URL points at a
    Postgres server, on Postgres too (each test in its own throwaway schema)."""
    if request.param == "sqlite":
        cat = SQLiteCatalog(settings.db_path)
        yield cat
        cat.close()
        return
    if not PG_URL:
        pytest.skip("set RADIODB_TEST_DATABASE_URL to run Postgres tests")
    import psycopg

    from radiodb.catalog.postgres import PostgresCatalog

    schema = f"test_{uuid.uuid4().hex[:12]}"
    cat = PostgresCatalog(PG_URL, schema=schema)
    yield cat
    cat.close()
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def db(settings, asr, catalog):
    d = RadioDB(
        settings,
        catalog=catalog,
        blobs=LocalBlobStore(settings.blob_dir),
        models=Models(vad=FakeVAD(), embedder=FakeEmbedder(), transcriber=asr, layers=default_layers()),
    )
    yield d


VOCAB = {
    "A": ["hello", "how", "is", "the", "budget", "looking"],
    "B": ["pretty", "good", "we", "saved", "money", "today"],
    "C": ["sorry", "i", "am", "late", "again", "folks"],
}
