"""Per-segment enrichment layers that run on top of diarized, transcribed segments.

Each layer gets the recording's 16 kHz mono audio plus its segments and returns, per segment,
``(data, vector)``: JSON-able features and/or an embedding. Layers are independent, so a new one
can be backfilled over existing recordings without re-running ASR or diarization.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

import numpy as np

from ..audio import MODEL_SR

LayerRow = tuple[dict | None, np.ndarray | None]


class Layer(Protocol):
    name: str
    producer: str

    def compute(self, audio16k: np.ndarray, segments: list[dict]) -> list[LayerRow]: ...


class TextEmbedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised embeddings, shape (len(texts), dim)."""
        ...


# ---- prosody ------------------------------------------------------------------------------

PAUSE_S = 0.25  # a gap between words at least this long counts as a pause
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def count_syllables(word: str) -> int:
    """English syllable estimate from vowel groups (good enough for a rate measure)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_VOWEL_GROUPS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(1, n)


def timing_features(words: list[dict], start_s: float, end_s: float) -> dict:
    """Speech/articulation rate and pauses, from ASR word timestamps."""
    dur = max(end_s - start_s, 1e-6)
    syl = sum(count_syllables(w["w"]) for w in words)
    gaps = [b["start"] - a["end"] for a, b in zip(words, words[1:])]
    pauses = [g for g in gaps if g >= PAUSE_S]
    speaking = max(dur - sum(pauses), 1e-6)
    return {
        "n_words": len(words),
        "speech_rate_sps": round(syl / dur, 3) if words else None,         # syllables / s, pauses included
        "articulation_rate_sps": round(syl / speaking, 3) if words else None,  # syllables / s, pauses removed
        "n_pauses": len(pauses),
        "pause_total_s": round(sum(pauses), 3),
        "pause_max_s": round(max(pauses), 3) if pauses else 0.0,
    }


def _semitones(hz: np.ndarray) -> np.ndarray:
    return 12 * np.log2(hz / 100.0)  # semitones re 100 Hz: comparable across speakers


class ProsodyLayer:
    """Pitch (F0), loudness, rate and pauses per segment, measured with Praat."""

    name = "prosody"

    def __init__(self, pitch_floor: float = 75.0, pitch_ceiling: float = 500.0):
        import parselmouth

        self._pm = parselmouth
        self.producer = f"praat-parselmouth/{parselmouth.__version__}"
        self.floor, self.ceiling = pitch_floor, pitch_ceiling

    def compute(self, audio16k: np.ndarray, segments: list[dict]) -> list[LayerRow]:
        return [(self._segment(audio16k, s), None) for s in segments]

    def _segment(self, a16: np.ndarray, seg: dict) -> dict | None:
        x = a16[int(seg["start_s"] * MODEL_SR): int(seg["end_s"] * MODEL_SR)].astype(np.float64)
        if len(x) < int(0.3 * MODEL_SR):
            return None
        snd = self._pm.Sound(x, sampling_frequency=MODEL_SR)
        out = timing_features(seg.get("words", []), seg["start_s"], seg["end_s"])

        f0 = snd.to_pitch(time_step=0.01, pitch_floor=self.floor, pitch_ceiling=self.ceiling)
        hz = f0.selected_array["frequency"]
        voiced = hz[hz > 0]
        out["voiced_frac"] = round(float(len(voiced) / max(len(hz), 1)), 3)
        if len(voiced) >= 5:
            st = _semitones(voiced)
            out.update(
                f0_median_hz=round(float(np.median(voiced)), 1),
                f0_mean_st=round(float(st.mean()), 2),
                f0_std_st=round(float(st.std()), 2),
                f0_range_st=round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
            )
        else:
            out.update(f0_median_hz=None, f0_mean_st=None, f0_std_st=None, f0_range_st=None)

        db = snd.to_intensity(minimum_pitch=self.floor).values.flatten()
        db = db[np.isfinite(db)]
        if len(db):
            # Energy-average in dB. Absolute level depends on mic gain, so compare within a recording.
            out["intensity_db"] = round(float(10 * math.log10(np.mean(10 ** (db / 10)))), 2)
        else:
            out["intensity_db"] = None
        return out


# ---- semantic -----------------------------------------------------------------------------

class SentenceEmbedder:
    def __init__(self, model: str, model_dir, device: str):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._m = SentenceTransformer(model, device=device, cache_folder=str(model_dir / "text"))

    def embed(self, texts: list[str]) -> np.ndarray:
        e = self._m.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        return e.astype(np.float32)


MIN_WORDS_FOR_SEMANTIC = 3


class SemanticLayer:
    """Text embedding of what was said, for search by meaning rather than exact words.

    A segment's vector is its own text embedding plus a down-weighted embedding of the
    preceding segment, so short replies ("yes, let's do that") keep some conversational
    context without outranking the turn that actually says the thing.
    """

    name = "semantic"

    def __init__(self, embedder: TextEmbedder, context_words: int = 25, context_weight: float = 0.35):
        self.embedder = embedder
        self.producer = f"{embedder.name}+ctx{context_weight}"
        self.context_words = context_words
        self.context_weight = context_weight

    def compute(self, audio16k: np.ndarray, segments: list[dict]) -> list[LayerRow]:
        idx = [i for i, s in enumerate(segments) if len(s["text"].split()) >= MIN_WORDS_FOR_SEMANTIC]
        out: list[LayerRow] = [(None, None)] * len(segments)
        if not idx:
            return out
        own = self.embedder.embed([segments[i]["text"] for i in idx])
        ctx_texts = [" ".join(segments[i - 1]["text"].split()[-self.context_words:]) if i else "" for i in idx]
        has_ctx = [j for j, t in enumerate(ctx_texts) if t]
        ctx = np.zeros_like(own)
        if has_ctx:
            ctx[has_ctx] = self.embedder.embed([ctx_texts[j] for j in has_ctx])
        vecs = own + self.context_weight * ctx
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        for i, v in zip(idx, vecs.astype(np.float32)):
            out[i] = (None, v)
        return out

    def embed_query(self, q: str) -> np.ndarray:
        return self.embedder.embed([q])[0]


# ---- emotion ------------------------------------------------------------------------------

class EmotionClassifier(Protocol):
    name: str
    labels: list[str]

    def probs(self, clip: np.ndarray) -> np.ndarray:
        """Class probabilities for one mono 16 kHz clip, aligned with ``labels``."""
        ...


_SUPERB_LABELS = {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad"}


class HubertEmotionClassifier:
    """``superb/hubert-base-superb-er`` (Apache-2.0): HuBERT-base fine-tuned on IEMOCAP, 4 classes.
    SUPERB reports ~65 % accuracy on IEMOCAP; treat labels as hints, not ground truth."""

    def __init__(self, model: str, model_dir, device: str):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        import os

        cache = str(model_dir / "emotion")
        # provenance: the hub id, or "local:<folder>" for a downloaded copy (not a machine path)
        self.name = f"local:{os.path.basename(os.path.normpath(model))}" if os.path.isdir(model) else model
        self._fe = AutoFeatureExtractor.from_pretrained(model, cache_dir=cache)
        self._m = AutoModelForAudioClassification.from_pretrained(model, cache_dir=cache).to(device).eval()
        self._device = device
        id2label = self._m.config.id2label
        self.labels = [_SUPERB_LABELS.get(id2label[i], id2label[i]) for i in range(len(id2label))]

    def probs(self, clip: np.ndarray) -> np.ndarray:
        import torch

        inputs = self._fe(clip, sampling_rate=MODEL_SR, return_tensors="pt")
        with torch.inference_mode():
            logits = self._m(**{k: v.to(self._device) for k, v in inputs.items()}).logits[0]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()


class EmotionLayer:
    """Categorical emotion per segment from the audio (not the words).

    Long segments are scored in windows of at most ``window_s`` and the class probabilities are
    averaged, weighted by window length. Stored: ``label`` (argmax), ``confidence`` (its
    probability) and all ``probs``, so callers can apply their own threshold.
    """

    name = "emotion"

    def __init__(self, classifier: EmotionClassifier, min_s: float = 1.0, window_s: float = 10.0):
        self.classifier = classifier
        self.producer = classifier.name
        self.min_s, self.window_s = min_s, window_s

    def compute(self, audio16k: np.ndarray, segments: list[dict]) -> list[LayerRow]:
        return [(self._segment(audio16k, s), None) for s in segments]

    def _segment(self, a16: np.ndarray, seg: dict) -> dict | None:
        if seg["end_s"] - seg["start_s"] < self.min_s:
            return None
        x = a16[int(seg["start_s"] * MODEL_SR): int(seg["end_s"] * MODEL_SR)]
        step = int(self.window_s * MODEL_SR)
        chunks = [x[i: i + step] for i in range(0, len(x), step)]
        chunks = [c for c in chunks if len(c) >= self.min_s * MODEL_SR] or chunks[:1]
        weights = np.array([len(c) for c in chunks], dtype=np.float64)
        p = sum(w * self.classifier.probs(c) for w, c in zip(weights, chunks)) / weights.sum()
        i = int(np.argmax(p))
        return {
            "label": self.classifier.labels[i],
            "confidence": round(float(p[i]), 3),
            "probs": {lab: round(float(v), 3) for lab, v in zip(self.classifier.labels, p)},
        }
