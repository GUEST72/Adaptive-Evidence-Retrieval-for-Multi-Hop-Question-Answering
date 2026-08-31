"""Disk-backed cache for LLM responses.

A 300-question sweep is ~25 minutes of waiting on the API and local compute is
under 3 seconds of it, so the only way to make iteration fast is to not make the
call at all. Re-running an unchanged config becomes a few seconds, and a sweep
that dies partway through resumes for free.

Backed by stdlib sqlite3: one file, atomic writes, no new dependency, and no
directory full of thousands of small JSON files.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

DEFAULT_CACHE_DIR = Path(".cache/llm")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key        TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    response   TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""

_connection: sqlite3.Connection | None = None
_connection_path: Path | None = None

hits = 0
misses = 0


def cache_dir() -> Path:
    """Cache location, overridable with LLM_CACHE_DIR (used by the tests)."""
    configured = os.getenv("LLM_CACHE_DIR")
    return Path(configured) if configured else DEFAULT_CACHE_DIR


def cache_key(
    *, model: str, prompt: str, max_tokens: int, temperature: float, provider: str = "groq"
) -> str:
    """Everything that can change the response goes into the key."""
    payload = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_connection() -> sqlite3.Connection:
    global _connection, _connection_path

    path = cache_dir() / "responses.db"
    if _connection is not None and _connection_path == path:
        return _connection

    if _connection is not None:
        _connection.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    _connection = sqlite3.connect(path)
    _connection.execute(_SCHEMA)
    _connection.commit()
    _connection_path = path
    return _connection


def get(key: str) -> str | None:
    global hits, misses

    row = _get_connection().execute(
        "SELECT response FROM responses WHERE key = ?", (key,)
    ).fetchone()

    if row is None:
        misses += 1
        return None

    hits += 1
    return row[0]


def put(key: str, response: str, model: str) -> None:
    connection = _get_connection()
    connection.execute(
        "INSERT OR REPLACE INTO responses (key, model, response, created_at) VALUES (?, ?, ?, ?)",
        (key, model, response, time.time()),
    )
    connection.commit()


def reset_stats() -> None:
    global hits, misses
    hits = misses = 0


def close() -> None:
    """Drop the handle so a later call reopens against the current cache_dir."""
    global _connection, _connection_path
    if _connection is not None:
        _connection.close()
    _connection = None
    _connection_path = None


def entry_count() -> int:
    return _get_connection().execute("SELECT COUNT(*) FROM responses").fetchone()[0]
