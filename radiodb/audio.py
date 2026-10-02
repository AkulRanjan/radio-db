"""Audio decoding, canonical storage encoding and slicing.

Decoding uses libsndfile (via soundfile): WAV, FLAC, AIFF, OGG/Vorbis, Opus and MP3.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MODEL_SR = 16_000

# Formats we re-encode to FLAC without losing anything. Anything else (MP3, Vorbis, Opus) is
# already lossy, so we keep the uploaded bytes untouched rather than transcode lossy -> lossless.
_LOSSLESS_FORMATS = {"WAV", "WAVEX", "FLAC", "AIFF", "W64", "RF64"}


class UnsupportedAudio(ValueError):
    pass


@dataclass
class Decoded:
    samples: np.ndarray  # float32, shape (frames, channels)
    sample_rate: int
    format: str
    subtype: str = ""

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def decode(data: bytes) -> Decoded:
    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            fmt, subtype = f.format, f.subtype
            samples = f.read(dtype="float32", always_2d=True)
            sr = f.samplerate
    except (sf.LibsndfileError, RuntimeError) as e:
        raise UnsupportedAudio(f"cannot decode audio: {e}") from e
    if samples.shape[0] == 0:
        raise UnsupportedAudio("audio is empty")
    return Decoded(samples, sr, fmt, subtype)


def canonical_bytes(data: bytes, decoded: Decoded) -> tuple[bytes, str]:
    """Bytes to store as the record and their file extension."""
    if decoded.format in _LOSSLESS_FORMATS:
        buf = io.BytesIO()
        # Match the source bit depth; FLAC tops out at 24-bit, so float/32-bit sources go to 24-bit.
        depth = "PCM_16" if decoded.subtype in ("PCM_16", "PCM_U8", "PCM_S8") else "PCM_24"
        sf.write(buf, decoded.samples, decoded.sample_rate, format="FLAC", subtype=depth)
        return buf.getvalue(), "flac"
    return data, decoded.format.lower()


def to_model_input(decoded: Decoded) -> np.ndarray:
    """Mono float32 at 16 kHz, the input every speech model here expects."""
    mono = decoded.samples.mean(axis=1)
    if decoded.sample_rate != MODEL_SR:
        g = gcd(decoded.sample_rate, MODEL_SR)
        mono = resample_poly(mono, MODEL_SR // g, decoded.sample_rate // g)
    return np.ascontiguousarray(mono, dtype=np.float32)


def slice_wav(decoded: Decoded, start_s: float, end_s: float) -> bytes:
    a = max(0, int(start_s * decoded.sample_rate))
    b = min(decoded.samples.shape[0], int(end_s * decoded.sample_rate))
    if b <= a:
        raise ValueError("empty time range")
    buf = io.BytesIO()
    sf.write(buf, decoded.samples[a:b], decoded.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
