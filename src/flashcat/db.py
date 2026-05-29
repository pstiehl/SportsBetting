"""SQLite ledger for events, source pulls, bets, weight history."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    league TEXT,
    home TEXT NOT NULL,
    away TEXT NOT NULL,
    commence_time TEXT NOT NULL,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS source_probs (
    event_id TEXT NOT NULL,
    source TEXT NOT NULL,
    home_win_prob REAL NOT NULL,
    captured_at TEXT NOT NULL,
    notes TEXT,
    PRIMARY KEY (event_id, source, captured_at)
);

CREATE TABLE IF NOT EXISTS lines (
    event_id TEXT NOT NULL,
    book TEXT NOT NULL,
    side TEXT NOT NULL,
    american INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    is_opening INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (event_id, book, side, captured_at)
);

CREATE TABLE IF NOT EXISTS results (
    event_id TEXT PRIMARY KEY,
    home_score INTEGER,
    away_score INTEGER,
    home_won INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bets (
    event_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    stake REAL NOT NULL,
    american_price INTEGER NOT NULL,
    won INTEGER,
    profit REAL,
    PRIMARY KEY (event_id, strategy)
);

CREATE TABLE IF NOT EXISTS weight_history (
    captured_at TEXT NOT NULL,
    source TEXT NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (captured_at, source)
);
"""


@contextmanager
def connect(path: Path | None = None):
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with connect(path) as c:
        c.executescript(SCHEMA)


def insert_weight_snapshot(weights: dict[str, float], captured_at: str) -> None:
    with connect() as c:
        for source, weight in weights.items():
            c.execute(
                "INSERT OR REPLACE INTO weight_history(captured_at, source, weight) VALUES (?, ?, ?)",
                (captured_at, source, float(weight)),
            )
