"""Cross-session search over past chat turns, backed by SQLite + FTS5.

Hermes Agent uses a SQLite database with an FTS5 virtual table to let the
agent recall what was discussed in previous sessions. We adopt the same
approach because it is:

* Zero-extra-dependency (Python stdlib ``sqlite3`` ships with FTS5 enabled
  on macOS / Linux / Windows wheels).
* Crash-safe (WAL journal mode).
* Fast enough for tens of thousands of turns on a laptop.
* Inspectable with any SQLite GUI / ``sqlite3`` CLI.

The ``recall_past`` agent tool is a thin wrapper around :func:`search`.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Iterable

from optiprofiler_agent.runtime import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    label      TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    ts         REAL    NOT NULL,
    role       TEXT    NOT NULL,
    content    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_ts      ON turns(ts);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    content,
    role        UNINDEXED,
    session_id  UNINDEXED,
    content='turns',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, content, role, session_id)
    VALUES (new.id, new.content, new.role, new.session_id);
END;

CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, content, role, session_id)
    VALUES ('delete', old.id, old.content, old.role, old.session_id);
END;
"""


@dataclass(frozen=True)
class TurnHit:
    session_id: str
    ts: float
    role: str
    content: str


def _connect() -> sqlite3.Connection:
    db = paths.session_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(_SCHEMA)
    try:
        conn.executescript(_FTS_SCHEMA)
    except sqlite3.OperationalError:
        # FTS5 unavailable on this build; LIKE-based search is the fallback.
        pass
    return conn


def _fts_available(conn: sqlite3.Connection) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='turns_fts'"
    )
    return cur.fetchone() is not None


def new_session(label: str | None = None) -> str:
    """Create a new session row, return its UUID."""
    sid = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, started_at, label) VALUES (?, ?, ?)",
            (sid, time.time(), label),
        )
    return sid


def log_turn(session_id: str, role: str, content: str) -> None:
    """Persist one user / assistant / tool turn. Best-effort, swallows errors."""
    if not content:
        return
    try:
        with _connect() as conn:
            # Auto-create the session row if the caller forgot to register it.
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, time.time()),
            )
            conn.execute(
                "INSERT INTO turns (session_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), role, content),
            )
    except sqlite3.Error:
        # Logging must never crash the chat loop.
        pass


def _escape_fts(query: str) -> str:
    """Quote every token so FTS5 syntax operators don't blow up on stray
    punctuation (``-``, ``"``, ``(``, ``)``, etc. are reserved)."""
    tokens = [t for t in query.replace('"', "").split() if t]
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def search(query: str, limit: int = 10) -> list[TurnHit]:
    """Search past turns. Falls back to LIKE if FTS5 isn't available."""
    query = (query or "").strip()
    if not query:
        return []
    with _connect() as conn:
        rows: Iterable[sqlite3.Row]
        if _fts_available(conn):
            fts_q = _escape_fts(query)
            if not fts_q:
                return []
            rows = conn.execute(
                """
                SELECT t.session_id, t.ts, t.role, t.content
                FROM turns_fts f
                JOIN turns t ON t.id = f.rowid
                WHERE turns_fts MATCH ?
                ORDER BY t.ts DESC
                LIMIT ?
                """,
                (fts_q, limit),
            ).fetchall()
        else:
            like = f"%{query}%"
            rows = conn.execute(
                """
                SELECT session_id, ts, role, content
                FROM turns
                WHERE content LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
    return [TurnHit(**dict(r)) for r in rows]


def list_sessions(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.started_at, s.label,
                   (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.session_id) AS turn_count
            FROM sessions s
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
