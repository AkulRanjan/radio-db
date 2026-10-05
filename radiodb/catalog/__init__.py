"""Relational catalog. SQLite for local development, PostgreSQL for production."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseCatalog, _iso, blob_to_vec, now_utc, vec_to_blob
from .sqlite import SQLiteCatalog

if TYPE_CHECKING:
    from ..config import Settings

Catalog = SQLiteCatalog  # backwards-compatible default

__all__ = [
    "BaseCatalog", "Catalog", "SQLiteCatalog", "open_catalog",
    "_iso", "blob_to_vec", "now_utc", "vec_to_blob",
]


def open_catalog(settings: "Settings") -> BaseCatalog:
    url = settings.database_url
    if url and url.startswith(("postgres://", "postgresql://")):
        from .postgres import PostgresCatalog

        return PostgresCatalog(url, schema=settings.database_schema or None)
    if url and not url.startswith("sqlite:"):
        raise ValueError(f"unsupported RADIODB_DATABASE_URL scheme: {url.split(':', 1)[0]}")
    path = url.removeprefix("sqlite:///") if url else settings.db_path
    return SQLiteCatalog(path)
