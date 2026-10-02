"""Content-addressed, write-once storage for audio.

Keys look like ``ab/abcdef...<sha256>.flac``. The local backend is for development; the S3
backend targets production (AWS S3, MinIO, R2, GCS interop — anything S3-compatible).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from .config import Settings


def blob_key(data: bytes, ext: str) -> tuple[str, str]:
    digest = hashlib.sha256(data).hexdigest()
    return digest, f"{digest[:2]}/{digest}.{ext}"


class BlobStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalBlobStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"invalid key: {key}")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        if p.exists():  # content-addressed: same key means same bytes
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class S3BlobStore:
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None):
        import boto3  # optional dependency: pip install radiodb[s3]

        if not bucket:
            raise ValueError("RADIODB_S3_BUCKET must be set for the s3 blob backend")
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)

    def put(self, key: str, data: bytes) -> None:
        self.s3.put_object(
            Bucket=self.bucket, Key=self.prefix + key, Body=data, ServerSideEncryption="AES256"
        )

    def get(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()

    def delete(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.bucket, Key=self.prefix + key)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.prefix + key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise


def make_blob_store(settings: Settings) -> BlobStore:
    if settings.blob_backend == "local":
        return LocalBlobStore(settings.blob_dir)
    if settings.blob_backend == "s3":
        return S3BlobStore(settings.s3_bucket, settings.s3_prefix, settings.s3_endpoint_url)
    raise ValueError(f"unknown blob backend: {settings.blob_backend}")
