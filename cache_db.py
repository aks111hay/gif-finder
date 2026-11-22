"""SQLite cache for embedding vectors and ranked search results.

Keeps vectors on disk instead of RAM (good for Render's 512 MB free tier) and
survives worker restarts. Every method swallows sqlite errors and acts like a
cache miss, so a broken cache can never crash a search.
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(os.environ.get(
    "CACHE_DB", Path(tempfile.gettempdir()) / "gif_finder_cache.db"))
MAX_VECTORS = 5000   # 5000 x 768 x 4 bytes = ~15 MB
MAX_RESULTS = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    key TEXT PRIMARY KEY, vec BLOB NOT NULL, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS results (
    key TEXT PRIMARY KEY, body TEXT NOT NULL, ts REAL NOT NULL);
"""


class CacheDB:
    def __init__(self, path: Path = DEFAULT_PATH, dim: int = 768):
        self.dim = dim
        self._lock = threading.Lock()
        self._writes = 0
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(SCHEMA)

    def _run(self, sql: str, params: tuple = (), fetch: bool = False):
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                row = cur.fetchone() if fetch else None
                self._conn.commit()
                return row
        except sqlite3.Error:
            return None

    # -- vectors ---------------------------------------------------------
    def get_vec(self, key: str):
        row = self._run("SELECT vec FROM vectors WHERE key=?", (key,), fetch=True)
        if not row or len(row[0]) != self.dim * 4:
            return None
        return np.frombuffer(row[0], dtype=np.float32).copy()

    def set_vec(self, key: str, vec: np.ndarray) -> None:
        self._run("INSERT OR REPLACE INTO vectors (key, vec, ts) VALUES (?,?,?)",
                  (key, np.asarray(vec, dtype=np.float32).tobytes(), time.time()))
        self._maybe_prune()

    # -- ranked results --------------------------------------------------
    def get_results(self, key: str, ttl: float):
        row = self._run("SELECT body, ts FROM results WHERE key=?", (key,), fetch=True)
        if not row or time.time() - row[1] > ttl:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set_results(self, key: str, matches: list) -> None:
        self._run("INSERT OR REPLACE INTO results (key, body, ts) VALUES (?,?,?)",
                  (key, json.dumps(matches), time.time()))
        self._maybe_prune()

    # -- housekeeping ----------------------------------------------------
    def _maybe_prune(self) -> None:
        self._writes += 1
        if self._writes % 100:
            return
        for table, limit in (("vectors", MAX_VECTORS), ("results", MAX_RESULTS)):
            self._run(f"DELETE FROM {table} WHERE key NOT IN "
                      f"(SELECT key FROM {table} ORDER BY ts DESC LIMIT ?)", (limit,))