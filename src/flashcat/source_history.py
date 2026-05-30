"""Persistent per-source prediction ledger.

This is the SQLite database Phil asked for: one row per (event, source) with
the pre-game probability, the realized outcome, the market closing prob, and
the market closing odds.

The blender / reweighter query this table over a rolling window (default 365
days) to compute Brier, log-loss, ROI, calibration slope, and CLV. That gives
us a single source of truth that's queryable from notebooks, the site
generator, and external tooling.

Schema
------
``predictions``
    event_id              TEXT
    sport                 TEXT
    source                TEXT
    commence_time         TEXT  (ISO 8601 UTC, walk-forward gate)
    home                  TEXT
    away                  TEXT
    home_prob             REAL  (source's pre-game prob of home winning, [0,1])
    home_won              INT   (0/1 outcome; NULL if not yet graded)
    market_close_home     REAL  (devigged market close prob, if available)
    market_close_decimal  REAL  (decimal odds taken on picked side)
    PRIMARY KEY (event_id, source)

``meta``
    sport, source, window_start, window_end, n_events, n_bets,
    brier, log_loss, accuracy, roi, calibration_slope, avg_clv_pp
    PRIMARY KEY (sport, source, window_end)
"""

from __future__ import annotations

import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import SOURCE_HISTORY_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    event_id TEXT NOT NULL,
    sport TEXT NOT NULL,
    source TEXT NOT NULL,
    commence_time TEXT NOT NULL,
    home TEXT,
    away TEXT,
    home_prob REAL NOT NULL,
    home_won INTEGER,
    market_close_home REAL,
    market_close_decimal REAL,
    PRIMARY KEY (event_id, source)
);
CREATE INDEX IF NOT EXISTS predictions_sport_source_time
    ON predictions(sport, source, commence_time);
CREATE INDEX IF NOT EXISTS predictions_commence
    ON predictions(commence_time);

CREATE TABLE IF NOT EXISTS meta (
    sport TEXT NOT NULL,
    source TEXT NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    n_events INTEGER,
    n_bets INTEGER,
    brier REAL,
    log_loss REAL,
    accuracy REAL,
    roi REAL,
    calibration_slope REAL,
    avg_clv_pp REAL,
    PRIMARY KEY (sport, source, window_end)
);
"""


@contextmanager
def connect(path: Path | None = None):
    p = path or SOURCE_HISTORY_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with connect(path) as c:
        c.executescript(SCHEMA)


def upsert_predictions(rows: Iterable[dict], path: Path | None = None) -> int:
    """Insert/replace prediction rows. Returns count written.

    Each row must include: event_id, sport, source, commence_time, home_prob.
    Optional: home, away, home_won, market_close_home, market_close_decimal.
    """
    init_db(path)
    n = 0
    with connect(path) as c:
        for r in rows:
            c.execute(
                """
                INSERT OR REPLACE INTO predictions
                  (event_id, sport, source, commence_time, home, away,
                   home_prob, home_won, market_close_home, market_close_decimal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["event_id"],
                    r["sport"],
                    r["source"],
                    r["commence_time"],
                    r.get("home"),
                    r.get("away"),
                    float(r["home_prob"]),
                    None if r.get("home_won") is None else int(bool(r["home_won"])),
                    None if r.get("market_close_home") is None else float(r["market_close_home"]),
                    None if r.get("market_close_decimal") is None else float(r["market_close_decimal"]),
                ),
            )
            n += 1
    return n


def query_window(
    sport: str | None = None,
    source: str | None = None,
    days: int = 365,
    *,
    end: date | None = None,
    path: Path | None = None,
) -> list[dict]:
    """Return graded predictions in the rolling window (default last 365 days)."""
    init_db(path)
    end = end or date.today()
    end_iso = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    # 365 days back from end
    start_dt = datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc)
    start_dt = start_dt.replace(day=start_dt.day) - _timedelta_days(days)
    start_iso = start_dt.isoformat()
    sql = """
        SELECT * FROM predictions
        WHERE home_won IS NOT NULL
          AND commence_time >= ?
          AND commence_time <= ?
    """
    args: list = [start_iso, end_iso]
    if sport:
        sql += " AND sport = ?"
        args.append(sport)
    if source:
        sql += " AND source = ?"
        args.append(source)
    with connect(path) as c:
        cur = c.execute(sql, args)
        return [dict(row) for row in cur.fetchall()]


def _timedelta_days(days: int):
    from datetime import timedelta
    return timedelta(days=days)


def upsert_meta(rows: Iterable[dict], path: Path | None = None) -> int:
    init_db(path)
    n = 0
    with connect(path) as c:
        for r in rows:
            c.execute(
                """
                INSERT OR REPLACE INTO meta
                  (sport, source, window_start, window_end, n_events, n_bets,
                   brier, log_loss, accuracy, roi, calibration_slope, avg_clv_pp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r["sport"],
                    r["source"],
                    r["window_start"],
                    r["window_end"],
                    r.get("n_events"),
                    r.get("n_bets"),
                    r.get("brier"),
                    r.get("log_loss"),
                    r.get("accuracy"),
                    r.get("roi"),
                    r.get("calibration_slope"),
                    r.get("avg_clv_pp"),
                ),
            )
            n += 1
    return n


# --- Stats helpers ---------------------------------------------------------


def _clip(p: float, lo: float = 1e-3, hi: float = 1 - 1e-3) -> float:
    return max(lo, min(hi, p))


def brier(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum((r["home_prob"] - r["home_won"]) ** 2 for r in rows) / len(rows)


def log_loss(rows: list[dict]) -> float | None:
    if not rows:
        return None
    s = 0.0
    for r in rows:
        p = _clip(r["home_prob"])
        y = r["home_won"]
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(rows)


def accuracy(rows: list[dict]) -> float | None:
    if not rows:
        return None
    hits = sum(1 for r in rows if (r["home_prob"] >= 0.5) == bool(r["home_won"]))
    return hits / len(rows)


def calibration_slope(rows: list[dict]) -> float | None:
    """Logistic-regression slope of outcome on logit(prob).

    β=1 is perfectly calibrated; β<1 means the source is too confident.
    Fitted via a small Newton step on Σ y log σ(α + β logit p) + (1-y) log (1 - σ).
    Returns None if the regression doesn't converge cleanly (e.g. <30 rows).
    """
    n = len(rows)
    if n < 30:
        return None
    xs = [math.log(_clip(r["home_prob"]) / (1 - _clip(r["home_prob"]))) for r in rows]
    ys = [float(r["home_won"]) for r in rows]
    alpha = 0.0
    beta = 1.0
    for _ in range(40):
        # gradients
        ga = 0.0
        gb = 0.0
        h_aa = 0.0
        h_ab = 0.0
        h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = 1.0 / (1.0 + math.exp(-(alpha + beta * x)))
            err = mu - y
            ga += err
            gb += err * x
            w = mu * (1 - mu)
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        d_alpha = (h_bb * ga - h_ab * gb) / det
        d_beta = (-h_ab * ga + h_aa * gb) / det
        alpha -= d_alpha
        beta -= d_beta
        if abs(d_alpha) + abs(d_beta) < 1e-7:
            break
    if not math.isfinite(beta):
        return None
    return beta


def avg_clv_pp(rows: list[dict]) -> float | None:
    """Mean closing-line value in percentage points (pre-game source prob − market close)."""
    vals = [
        100.0 * (r["home_prob"] - r["market_close_home"])
        for r in rows
        if r.get("market_close_home") is not None
    ]
    if not vals:
        return None
    return sum(vals) / len(vals)


def roi_flat(rows: list[dict]) -> tuple[float | None, int, int]:
    """ROI assuming flat $100 stake on the higher-prob side at market_close_decimal.

    Returns (roi, n_bets, n_wins). Skips rows missing market_close_decimal.
    """
    wagered = 0.0
    profit = 0.0
    wins = 0
    n_bets = 0
    for r in rows:
        dec = r.get("market_close_decimal")
        if dec is None or dec <= 1.0:
            continue
        pick_home = r["home_prob"] >= 0.5
        won = (pick_home and r["home_won"]) or (not pick_home and not r["home_won"])
        n_bets += 1
        wagered += 100.0
        if won:
            profit += 100.0 * (dec - 1.0)
            wins += 1
        else:
            profit -= 100.0
    if wagered <= 0:
        return None, 0, 0
    return profit / wagered, n_bets, wins
