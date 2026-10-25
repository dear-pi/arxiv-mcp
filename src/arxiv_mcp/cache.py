"""A tiny SQLite-backed TTL cache.

arXiv asks clients to keep request volume low, so caching responses is both
politeness and latency. The cache stores raw response bodies keyed by the
full request URL, which keeps the key logic trivially correct.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    expires_at REAL NOT NULL
)
"""


class TTLCache:
    """Thread-safe, process-local cache with a per-entry time-to-live."""

    def __init__(self, path: str | Path, ttl_seconds: float = 24 * 3600) -> None:
        self._path = str(path)
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=10)

    def get(self, key: str) -> str | None:
        """Return the cached value, or None when absent or expired."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at <= time.time():
            self.delete(key)
            return None
        return value

    def set(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entries (key, value, expires_at) VALUES (?, ?, ?)",
                (key, value, time.time() + self._ttl),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM entries WHERE key = ?", (key,))

    def clear(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM entries")

    def __len__(self) -> int:
        with self._lock, self._connect() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE expires_at > ?", (time.time(),)
            ).fetchone()
        return int(count)
