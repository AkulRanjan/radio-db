"""Runtime configuration, read from environment variables (prefix RADIODB_)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"RADIODB_{name}", default)


@dataclass(frozen=True)
class Settings:
    home: Path = field(default_factory=lambda: Path(_env("HOME", "./data")).resolve())

    # Blob storage: "local" for development, "s3" for production (any S3-compatible store).
    blob_backend: str = field(default_factory=lambda: _env("BLOB_BACKEND", "local"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env("S3_PREFIX", "blobs/"))
    s3_endpoint_url: str | None = field(default_factory=lambda: os.environ.get("RADIODB_S3_ENDPOINT_URL"))

    retention_days: int = field(default_factory=lambda: int(_env("RETENTION_DAYS", "365")))

    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cuda | cpu
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "small.en"))

    # Cosine similarity between ECAPA embeddings at or above which two voices are the same person.
    speaker_match_threshold: float = field(default_factory=lambda: float(_env("SPEAKER_MATCH_THRESHOLD", "0.60")))
    # Agglomerative clustering distance threshold (1 - cosine) used during diarization.
    diarization_threshold: float = field(default_factory=lambda: float(_env("DIARIZATION_THRESHOLD", "0.6")))
    # A voice needs at least this much speech in a recording to count as its own speaker.
    min_speaker_s: float = field(default_factory=lambda: float(_env("MIN_SPEAKER_S", "3.0")))

    @property
    def db_path(self) -> Path:
        return self.home / "catalog.sqlite3"

    @property
    def blob_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def model_dir(self) -> Path:
        return self.home / "models"
