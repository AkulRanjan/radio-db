"""Real model backends. Heavy imports are deferred so the package loads without ML deps.

All three are non-gated (no Hugging Face token needed):
  - Silero VAD
  - SpeechBrain ECAPA-TDNN speaker encoder (192-d, trained on VoxCeleb)
  - faster-whisper (CTranslate2 Whisper) with word timestamps, English-only models by default
"""

from __future__ import annotations

import logging
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
    name = "speechbrain/spkrec-ecapa-voxceleb"
    dim = 192

    def __init__(self, model_dir: Path, device: str, batch_size: int = 32):
        from speechbrain.inference.speaker import EncoderClassifier

        kwargs = {}
        try:  # Windows without developer mode cannot create the symlinks SpeechBrain uses by default
            from speechbrain.utils.fetching import LocalStrategy

            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self._model = EncoderClassifier.from_hparams(
            source=self.name, savedir=str(model_dir / "ecapa"), run_opts={"device": device}, **kwargs
        )
        self._device = device
        self._bs = batch_size

    def embed(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        out = []
        for i in range(0, len(clips), self._bs):
            batch = clips[i: i + self._bs]
            n = max(len(c) for c in batch)
            wavs = torch.zeros(len(batch), n)
            lens = torch.zeros(len(batch))
            for j, c in enumerate(batch):
                wavs[j, : len(c)] = torch.from_numpy(c)
                lens[j] = len(c) / n
            with torch.inference_mode():
                e = self._model.encode_batch(wavs.to(self._device), lens.to(self._device))
            out.append(e.squeeze(1).float().cpu().numpy())
        emb = np.concatenate(out).astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)


class WhisperTranscriber:
    def __init__(self, model: str, model_dir: Path, device: str):
        from faster_whisper import WhisperModel

        self.name = f"faster-whisper/{model}"
        compute = "float16" if device == "cuda" else "int8"
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
        segments, _info = self._model.transcribe(
            audio16k, language="en", word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, beam_size=5,
        )
        words = [
            Word(w.word, float(w.start), float(w.end), float(w.probability))
            for seg in segments for w in (seg.words or [])
        ]
        return Transcript(words=words, model=self.name)
