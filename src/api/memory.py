"""SQLite conversation memory and document registry for RagFlow.

Multi user by design: every turn is stored under a session id. Naive RAG stores
history but does not use it to reshape the query. Reformulating a question from
prior turns is a 2023 advanced technique, so it is intentionally absent here.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List

_DB_PATH = "ragflow.sqlite3"


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
        c.execute("""CREATE TABLE IF NOT EXISTS documents (
                file_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                chunks INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )""")
        # Every history read filters on session_id, which is a full scan of every
        # session's turns without this.
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_logs_session "
            "ON chat_logs (session_id, id DESC)"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_turn(session_id: str, question: str, answer: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_logs (session_id, question, answer, created_at) "
            "VALUES (?,?,?,?)",
            (session_id, question, answer, _now()),
        )


def get_history(session_id: str, limit: int = 10) -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT question, answer FROM chat_logs WHERE session_id=? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def register_document(file_id: str, filename: str, chunks: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (file_id, filename, chunks, created_at) "
            "VALUES (?,?,?,?)",
            (file_id, filename, chunks, _now()),
        )


def list_documents() -> List[Dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT file_id, filename, chunks, created_at FROM documents "
            "ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def remove_document(file_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE file_id=?", (file_id,))
