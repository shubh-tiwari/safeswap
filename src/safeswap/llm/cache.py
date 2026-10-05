"""SQLite memo of model calls, so re-running an experiment costs nothing.

Keyed by a hash of (model, messages, params). This is experiment plumbing, not the semantic
cache under study (that one lives in policies.py and is part of what gets measured).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def key(model: str, messages: list[dict], params: dict[str, Any]) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CallCache:
    def __init__(self, path: str | Path = ".cache/calls.sqlite"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()

    def get(self, k: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM calls WHERE k = ?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO calls (k, v) VALUES (?, ?)", (k, json.dumps(v))
            )
            self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
