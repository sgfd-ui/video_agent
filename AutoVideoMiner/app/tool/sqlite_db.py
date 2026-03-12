"""SQLite schema and atomic persistence helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

DEFAULT_LEVELS = ("普通行为", "威胁行为", "需要关注的行为")


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level_name TEXT UNIQUE NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            level_id INTEGER NOT NULL,
            vector_id TEXT,
            full_description TEXT,
            FOREIGN KEY(level_id) REFERENCES event_levels(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            keyword TEXT UNIQUE NOT NULL,
            score REAL,
            reason TEXT,
            numer INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS visited_urls (
            url TEXT PRIMARY KEY,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    for level in DEFAULT_LEVELS:
        cur.execute("INSERT OR IGNORE INTO event_levels(level_name) VALUES (?)", (level,))

    conn.commit()
    conn.close()


def fetch_event_snapshot(db_path: str) -> list[str]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT event_name FROM event_categories ORDER BY id")
    rows = [row[0] for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_search_history_keywords(db_path: str) -> set[str]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT keyword FROM search_history")
    rows = {row[0] for row in cur.fetchall()}
    conn.close()
    return rows


def upsert_search_history(db_path: str, platform: str, keyword: str, score: float, reason: str) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO search_history(platform, keyword, score, reason)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(keyword)
        DO UPDATE SET score=excluded.score, reason=excluded.reason, platform=excluded.platform
        """,
        (platform, keyword, score, reason),
    )
    conn.commit()
    conn.close()


def update_search_numer(db_path: str, keyword: str, numer: int) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE search_history SET numer=? WHERE keyword=?", (numer, keyword))
    conn.commit()
    conn.close()


def add_visited_urls(db_path: str, urls: Iterable[str]) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.executemany("INSERT OR IGNORE INTO visited_urls(url) VALUES (?)", [(u,) for u in urls])
    conn.commit()
    conn.close()


def is_url_visited(db_path: str, url: str) -> bool:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM visited_urls WHERE url=? LIMIT 1", (url,))
    hit = cur.fetchone() is not None
    conn.close()
    return hit
