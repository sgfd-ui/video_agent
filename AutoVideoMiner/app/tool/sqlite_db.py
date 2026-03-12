"""SQLite schema and atomic persistence helpers."""

from __future__ import annotations

import sqlite3
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from AutoVideoMiner.app.core.logger import get_logger

LOGGER = get_logger("tool.sqlite_db")
DEFAULT_LEVELS = ("普通行为", "威胁行为", "需要关注的行为")


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def _migrate_search_history(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='search_history'")
    row = cur.fetchone()
    if not row or not row[0]:
        return
    sql_text: str = row[0].lower()
    need_rebuild = "keyword text unique" in sql_text or "create_time" not in sql_text
    if not need_rebuild:
        return

    LOGGER.info("Migrating search_history schema to composite unique(platform, keyword) + create_time")
    cur.execute("ALTER TABLE search_history RENAME TO search_history_old")
    cur.execute(
        """
        CREATE TABLE search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            platform TEXT NOT NULL,
            keyword TEXT NOT NULL,
            score REAL,
            reason TEXT,
            numer INTEGER DEFAULT 0,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform, keyword)
        )
        """
    )
    cur.execute(
        """
        INSERT INTO search_history(platform, keyword, score, reason, numer, create_time)
        SELECT platform, keyword, score, reason, COALESCE(numer,0), CURRENT_TIMESTAMP
        FROM search_history_old
        """
    )
    cur.execute("DROP TABLE search_history_old")


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
            keyword TEXT NOT NULL,
            score REAL,
            reason TEXT,
            numer INTEGER DEFAULT 0,
            create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(platform, keyword)
        )
        """
    )
    _migrate_search_history(cur)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_search_history_platform_keyword ON search_history(platform, keyword)")

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
    LOGGER.info("Database initialized: %s", db_path)


def fetch_event_snapshot(db_path: str) -> list[str]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT event_name FROM event_categories ORDER BY id")
    rows = [row[0] for row in cur.fetchall()]
    conn.close()
    return rows


def fetch_search_history_keywords(db_path: str, platform: str | None = None) -> set[str]:
    conn = _connect(db_path)
    cur = conn.cursor()
    if platform:
        cur.execute("SELECT keyword FROM search_history WHERE platform=?", (platform,))
    else:
        cur.execute("SELECT keyword FROM search_history")
    rows = {row[0] for row in cur.fetchall()}
    conn.close()
    return rows


def fetch_search_records_exact(db_path: str, tasks: list[dict[str, str]]) -> list[dict]:
    if not tasks:
        return []
    conn = _connect(db_path)
    cur = conn.cursor()
    records: list[dict] = []
    for task in tasks:
        cur.execute(
            """
            SELECT platform, keyword, score, reason, numer, create_time
            FROM search_history
            WHERE platform=? AND keyword=?
            """,
            (task["platform"], task["keyword"]),
        )
        row = cur.fetchone()
        if row:
            records.append(
                {
                    "platform": row[0],
                    "keyword": row[1],
                    "score": row[2],
                    "reason": row[3],
                    "numer": row[4],
                    "create_time": row[5],
                }
            )
    conn.close()
    return records


def fetch_search_records_similar(db_path: str, platform: str, keyword: str, threshold: float = 0.8) -> list[dict]:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT platform, keyword, score, reason, numer, create_time
        FROM search_history
        WHERE platform=?
        ORDER BY create_time DESC
        LIMIT 100
        """,
        (platform,),
    )
    rows = cur.fetchall()
    conn.close()

    records = []
    for row in rows:
        sim = SequenceMatcher(None, keyword.lower(), (row[1] or "").lower()).ratio()
        if sim > threshold:
            records.append(
                {
                    "platform": row[0],
                    "keyword": row[1],
                    "score": row[2],
                    "reason": row[3],
                    "numer": row[4],
                    "create_time": row[5],
                    "similarity": round(sim, 3),
                }
            )
    return records


def upsert_search_history(db_path: str, platform: str, keyword: str, score: float, reason: str) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO search_history(platform, keyword, score, reason, create_time)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(platform, keyword)
        DO UPDATE SET score=excluded.score, reason=excluded.reason, create_time=CURRENT_TIMESTAMP
        """,
        (platform, keyword, score, reason),
    )
    conn.commit()
    conn.close()


def update_search_numer(db_path: str, platform: str, keyword: str, numer: int) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("UPDATE search_history SET numer=?, create_time=CURRENT_TIMESTAMP WHERE platform=? AND keyword=?", (numer, platform, keyword))
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
