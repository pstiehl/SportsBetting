"""source_history.db CRUD + stats tests."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flashcat import source_history as sh


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    p = tmp_path / "source_history.db"
    monkeypatch.setattr(sh, "SOURCE_HISTORY_DB_PATH", p)
    sh.init_db(p)
    return p


def _row(eid: str, sport: str, source: str, days_ago: int,
         prob: float, won: bool, close: float | None = None) -> dict:
    return {
        "event_id": eid,
        "sport": sport,
        "source": source,
        "commence_time": (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(),
        "home": "h",
        "away": "a",
        "home_prob": prob,
        "home_won": won,
        "market_close_home": close,
        "market_close_decimal": (1 / close) if close else None,
    }


def test_upsert_and_query(fake_db):
    rows = [_row(f"e{i}", "mlb", "src", i + 1, 0.6, i % 2 == 0) for i in range(10)]
    n = sh.upsert_predictions(rows, path=fake_db)
    assert n == 10
    got = sh.query_window(sport="mlb", source="src", days=365, path=fake_db)
    assert len(got) == 10


def test_brier_and_log_loss(fake_db):
    rng = random.Random(1)
    rows = [_row(f"e{i}", "mlb", "src", 1, 0.5 + rng.uniform(-0.1, 0.1),
                 rng.random() < 0.55) for i in range(200)]
    sh.upsert_predictions(rows, path=fake_db)
    q = sh.query_window(sport="mlb", source="src", days=365, path=fake_db)
    b = sh.brier(q)
    ll = sh.log_loss(q)
    assert 0.0 < b < 0.30
    assert ll is not None and ll > 0.0


def test_calibration_slope_near_one_for_calibrated_data(fake_db):
    rng = random.Random(0)
    rows = []
    for i in range(500):
        p = rng.uniform(0.1, 0.9)
        y = rng.random() < p
        rows.append(_row(f"e{i}", "mlb", "src", 1, p, y))
    sh.upsert_predictions(rows, path=fake_db)
    q = sh.query_window(sport="mlb", source="src", path=fake_db)
    slope = sh.calibration_slope(q)
    assert slope is not None
    assert 0.6 < slope < 1.6


def test_window_filter_excludes_old(fake_db):
    rows = [_row("e1", "mlb", "src", 800, 0.6, True)]
    sh.upsert_predictions(rows, path=fake_db)
    q = sh.query_window(sport="mlb", source="src", days=365, path=fake_db)
    assert len(q) == 0
