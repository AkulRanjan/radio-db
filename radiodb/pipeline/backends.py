"""Real model backends. Heavy imports are deferred so the package loads without ML deps.

All three are non-gated (no Hugging Face token needed):
  - Silero VAD
  - SpeechBrain ECAPA-TDNN speaker encoder (192-d, trained on VoxCeleb)
  - faster-whisper (CTranslate2 Whisper) with word timestamps, English-only models by default
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from ..audio import MODEL_SR
from .interfaces import Span, Transcript, Word

log = logging.getLogger(__name__)


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SileroVAD:
    name = "silero-vad"

    def __init__(self, min_speech_ms: int = 250, min_silence_ms: int = 300):
        from silero_vad import load_silero_vad

        self._model = load_silero_vad()
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms

    def speech(self, audio16k: np.ndarray) -> list[Span]:
        import torch
        from silero_vad import get_speech_timestamps

        ts = get_speech_timestamps(
            torch.from_numpy(audio16k), self._model, sampling_rate=MODEL_SR,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms, return_seconds=True,
        )
        return [Span(float(t["start"]), float(t["end"])) for t in ts]


class EcapaEmbedder:
    """ECAPA-TDNN speaker embeddings.

    Memory-safe for any input: clips are sorted by length and batched so the *padded* audio per
    batch stays under ``max_batch_s``, and clips longer than ``max_clip_s`` are embedded in
    pieces whose embeddings are averaged. (Batching 32 long clips once needed ~12 GB on a 6 GB GPU.)
    """

    name = "speechbrain/spkrec-ecapa-voxceleb"
    dim = 192

    def __init__(self, model_dir: Path, device: str, max_batch_s: float = 120.0, max_clip_s: float = 30.0):
        from speechbrain.inference.speaker import EncoderClassifier

        kwargs = {}
        try:  # Windows without developer mode cannot create the symlinks SpeechBrain uses by default
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self._device = "cuda:0" if device == "cuda" else device  # SpeechBrain wants an explicit index
        self._model = EncoderClassifier.from_hparams(
            source=self.name, savedir=str(model_dir / "ecapa"), run_opts={"device": self._device}, **kwargs
        )
        self.max_batch = int(max_batch_s * MODEL_SR)
        self.max_clip = int(max_clip_s * MODEL_SR)

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        # split long clips; remember which pieces belong to which clip
        pieces, owner = [], []
        for i, c in enumerate(clips):
            parts = [c[j: j + self.max_clip] for j in range(0, max(len(c), 1), self.max_clip)]
            parts = [p for p in parts if len(p) >= MODEL_SR // 2] or parts[:1]
            pieces += parts
            owner += [i] * len(parts)
        vecs = self._embed_pieces(pieces)
        out = np.zeros((len(clips), self.dim), dtype=np.float32)
        weights = np.zeros(len(clips))
        for v, i, p in zip(vecs, owner, pieces):
            out[i] += v * len(p)
            weights[i] += len(p)
        out /= np.maximum(weights, 1)[:, None]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def _embed_pieces(self, pieces: list[np.ndarray]) -> np.ndarray:
        import torch

        order = sorted(range(len(pieces)), key=lambda k: len(pieces[k]))
        result = np.zeros((len(pieces), self.dim), dtype=np.float32)
        i = 0
        while i < len(order):
            # grow the batch while padded size (count x longest) fits the budget
            j = i + 1
            while j < len(order) and (j + 1 - i) * len(pieces[order[j]]) <= self.max_batch:
                j += 1
            idx = order[i:j]
            n = max(len(pieces[k]) for k in idx)
            wavs = torch.zeros(len(idx), n)
            lens = torch.zeros(len(idx))
            for r, k in enumerate(idx):
                wavs[r, : len(pieces[k])] = torch.from_numpy(np.ascontiguousarray(pieces[k]))
                lens[r] = len(pieces[k]) / n
            with torch.inference_mode():
                e = self._model.encode_batch(wavs.to(self._device), lens.to(self._device))
            result[idx] = e.squeeze(1).float().cpu().numpy()
            i = j
        return result


class WhisperTranscriber:
    def __init__(self, model: str, model_dir: Path, device: str, compute: str | None = None):
        from faster_whisper import WhisperModel

        from .asr_process import ensure_cuda_dlls

        if device == "cuda":
            ensure_cuda_dlls()
        self.name = f"faster-whisper/{model}"
        compute = compute or ("float16" if device == "cuda" else "int8")
        try:
            self._model = WhisperModel(
                model, device=device, compute_type=compute, download_root=str(model_dir / "whisper")
            )
        except (RuntimeError, ValueError) as e:  # e.g. missing cuDNN/cuBLAS DLLs
            if device == "cpu":
                raise
            log.warning("faster-whisper on %s failed (%s); falling back to CPU", device, e)
            self._model = WhisperModel(
                model, device="cpu", compute_type="int8", download_root=str(model_dir / "whisper")
            )

    def transcribe(self, audio16k: np.ndarray) -> Transcript:
        # torch's oneDNN convolutions (ECAPA, run just before) switch the CPU to flush-denormals-
        # to-zero and leave it on. CTranslate2's int8 CPU kernels then compute slightly
        # differently, which measurably degraded transcripts ("a beep" -> "of deep"). Reset it.
        if "torch" in sys.modules:
            sys.modules["torch"].set_flush_denormal(False)
        segments, _info = self._model.transcribe(
            audio16k, language="en", word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, beam_size=5,
        )
        words = [
            Word(w.word, float(w.start), float(w.end), float(w.probability))
            for seg in segments for w in (seg.words or [])
        ]
        return Transcript(words=words, model=self.name)
