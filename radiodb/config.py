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

    # Catalog: empty -> SQLite at <home>/catalog.sqlite3; "postgresql://user:pass@host/db" in production.
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))
    database_schema: str = field(default_factory=lambda: _env("DATABASE_SCHEMA", ""))

    # Blob storage: "local" for development, "s3" for production (any S3-compatible store).
    blob_backend: str = field(default_factory=lambda: _env("BLOB_BACKEND", "local"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    s3_prefix: str = field(default_factory=lambda: _env("S3_PREFIX", "blobs/"))
    s3_endpoint_url: str | None = field(default_factory=lambda: os.environ.get("RADIODB_S3_ENDPOINT_URL"))

    job_max_attempts: int = field(default_factory=lambda: int(_env("JOB_MAX_ATTEMPTS", "3")))
    # Run a worker thread inside the API process (local dev). Production: run `radiodb worker`.
    inline_worker: bool = field(default_factory=lambda: _env("INLINE_WORKER", "0") == "1")

    # API authentication: "header" (trust X-User-Id; local dev only) or "jwt" (verified bearer tokens).
    auth_mode: str = field(default_factory=lambda: _env("AUTH", "header"))
    jwt_secret: str = field(default_factory=lambda: _env("JWT_SECRET", ""))
    jwt_jwks_url: str = field(default_factory=lambda: _env("JWT_JWKS_URL", ""))
    jwt_audience: str = field(default_factory=lambda: _env("JWT_AUDIENCE", ""))
    jwt_issuer: str = field(default_factory=lambda: _env("JWT_ISSUER", ""))
    jwt_user_claim: str = field(default_factory=lambda: _env("JWT_USER_CLAIM", "sub"))

    # Limits: largest accepted upload, and total audio hours per user (0 = unlimited).
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "1024")))
    user_quota_hours: float = field(default_factory=lambda: float(_env("USER_QUOTA_HOURS", "0")))

    retention_days: int = field(default_factory=lambda: int(_env("RETENTION_DAYS", "365")))

    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))  # auto | cuda | cpu
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "small.en"))
    # Whisper precision: "" = float16 on GPU, int8 on CPU. On 10 min of AMI, CPU int8 vs float32 WER was
    # 17.0 % vs 16.9 % at half the time, so int8 is the CPU default; set "float32" to trade speed for that.
    asr_compute: str = field(default_factory=lambda: _env("ASR_COMPUTE", ""))
    # Run Whisper in its own process: "auto" = on CPU (avoids torch/CTranslate2 OpenMP interference), "on", "off".
    asr_isolate: str = field(default_factory=lambda: _env("ASR_ISOLATE", "auto"))
    text_model: str = field(default_factory=lambda: _env("TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    # Enrichment layers run after diarization + ASR (comma-separated; empty to disable all).
    layers: tuple[str, ...] = field(
        default_factory=lambda: tuple(x.strip() for x in _env("LAYERS", "prosody,semantic,emotion").split(",") if x.strip())
    )
    emotion_model: str = field(default_factory=lambda: _env("EMOTION_MODEL", "superb/hubert-base-superb-er"))
    # An emotion label only counts (search filter, tags, profile) at or above this probability.
    emotion_min_confidence: float = field(default_factory=lambda: float(_env("EMOTION_MIN_CONFIDENCE", "0.5")))

    # Defaults calibrated on 4.1 h of AMI meetings (8 recordings, 4 speakers each; see architecture.md
    # §12.13). Re-run `radiodb calibrate` on your own recordings to confirm.
    # Cosine similarity between ECAPA embeddings at or above which two voices are the same person.
    speaker_match_threshold: float = field(default_factory=lambda: float(_env("SPEAKER_MATCH_THRESHOLD", "0.58")))
    # Agglomerative clustering distance threshold (1 - cosine) used during diarization.
    diarization_threshold: float = field(default_factory=lambda: float(_env("DIARIZATION_THRESHOLD", "0.70")))
    # A voice needs at least this much speech in a recording to count as its own speaker
    # (capped at 10 % of the recording's speech, so short calls keep brief speakers).
    min_speaker_s: float = field(default_factory=lambda: float(_env("MIN_SPEAKER_S", "20")))

    @property
    def db_path(self) -> Path:
        return self.home / "catalog.sqlite3"

    @property
    def blob_dir(self) -> Path:
        return self.home / "blobs"

    @property
    def model_dir(self) -> Path:
        return self.home / "models"
