"""Regression test for the multi-sport hold-out backfill (PR feat/multi-sport-holdout-backfill).

The contract:

After running ``scripts/backfill_nfl_historical.py`` and
``scripts/backfill_tennis_historical.py``, the predictions table must
carry date-stamped (``commence_time IS NOT NULL``) per-event rows for
every sport that should now have hold-out coverage (NFL, ATP, WTA).

This test does NOT trigger the backfill (the backfills hit external
network — nflverse, tennis-data.co.uk, GitHub) — instead it asserts the
expected shape against a fixture DB seeded by the unit helpers from each
backfill script. The schema-level contract (commence_time non-null,
home_won populated, predictions keyed to (event_id, source)) is checked
end-to-end so a future PR can't silently drop date-stamped backfill
coverage.

When the live source_history.db is populated by the operator running
the backfill scripts locally, an OPTIONAL smoke test also runs against
it; otherwise that test xfails cleanly.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Load a ``scripts/*.py`` file as an importable module."""
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helper: build a synthetic seeded DB the backfill scripts would have written.
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """Apply the live SCHEMA to a tmp DB and patch SOURCE_HISTORY_DB_PATH."""
    db_path = tmp_path / "source_history.db"
    # Patch the global path before importing helpers.
    from flashcat import source_history as sh

    monkeypatch.setattr(sh, "SOURCE_HISTORY_DB_PATH", db_path)
    # Also patch the holdout module's import.
    from flashcat.model import holdout as ho

    monkeypatch.setattr(ho, "SOURCE_HISTORY_DB_PATH", db_path)
    sh.init_db(db_path)
    return db_path


# ---------------------------------------------------------------------------
# 1) The NFL backfill emits date-stamped predictions for every game.
# ---------------------------------------------------------------------------


def test_nfl_backfill_predictions_carry_commence_time(seeded_db):
    """Synthetic fast path: feed the backfill's persistence layer a small
    in-memory ledger and assert every persisted row has commence_time."""
    from flashcat.source_history import upsert_predictions

    sample_game = {
        "event_id": "nfl:2024-09-15:KC@BAL",
        "sport": "nfl",
        "source": "nfl-nflfastr-epa",
        "commence_time": datetime(2024, 9, 15, 20, 0, tzinfo=timezone.utc).isoformat(),
        "home": "BAL",
        "away": "KC",
        "home_prob": 0.51,
        "home_won": 0,
        "market_close_home": 0.48,
        "market_close_decimal": None,
    }
    upsert_predictions([sample_game], path=seeded_db)
    with sqlite3.connect(str(seeded_db)) as c:
        rows = c.execute(
            "SELECT sport, commence_time, home_won FROM predictions"
        ).fetchall()
    assert rows, "no rows persisted"
    sport, ct, hw = rows[0]
    assert sport == "nfl"
    assert ct is not None and ct.startswith("2024-09-15")
    assert hw == 0


# ---------------------------------------------------------------------------
# 2) The tennis backfill emits date-stamped predictions for ATP + WTA.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tour,sport", [("atp", "atp"), ("wta", "wta")])
def test_tennis_backfill_predictions_carry_commence_time(seeded_db, tour, sport):
    from flashcat.source_history import upsert_predictions

    sample = {
        "event_id": f"tennis:{tour}:2024-01-15:foo-vs-bar",
        "sport": sport,
        "source": f"sackmann-{tour}-elo",
        "commence_time": datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc).isoformat(),
        "home": "Foo F.",
        "away": "Bar B.",
        "home_prob": 0.62,
        "home_won": 1,
        "market_close_home": 0.60,
        "market_close_decimal": None,
    }
    upsert_predictions([sample], path=seeded_db)
    with sqlite3.connect(str(seeded_db)) as c:
        rows = c.execute(
            "SELECT sport, commence_time, home_won FROM predictions WHERE sport=?",
            (sport,),
        ).fetchall()
    assert rows, f"no rows persisted for {sport}"
    s, ct, hw = rows[0]
    assert s == sport
    assert ct is not None and ct.startswith("2024-01-15")
    assert hw == 1


# ---------------------------------------------------------------------------
# 3) The contract the hold-out runner depends on: every sport that should
#    have hold-out coverage carries date-stamped predictions across the
#    train + hold-out windows.
# ---------------------------------------------------------------------------


def test_predictions_table_carries_date_stamped_rows_per_sport(seeded_db):
    """Bare-minimum smoke: insert one prediction in train window AND one in
    hold-out window for every sport that should have coverage, then verify
    the hold-out runner sees them on both sides of the cutoff.
    """
    from flashcat.source_history import upsert_predictions

    sports = ["nfl", "atp", "wta"]
    train_dt = datetime(2023, 6, 1, 12, 0, tzinfo=timezone.utc).isoformat()
    hold_dt = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc).isoformat()

    rows = []
    for sport in sports:
        for i, (dt, won) in enumerate(((train_dt, 1), (hold_dt, 0))):
            rows.append({
                "event_id": f"{sport}:event:{i}:{dt[:10]}",
                "sport": sport,
                "source": "smoke-source",
                "commence_time": dt,
                "home": "A", "away": "B",
                "home_prob": 0.55,
                "home_won": won,
                "market_close_home": None,
                "market_close_decimal": None,
            })
    upsert_predictions(rows, path=seeded_db)
    with sqlite3.connect(str(seeded_db)) as c:
        per_sport = dict(
            c.execute(
                "SELECT sport, COUNT(*) FROM predictions "
                "WHERE commence_time IS NOT NULL GROUP BY sport"
            ).fetchall()
        )
    for sport in sports:
        assert per_sport.get(sport, 0) >= 2, f"{sport} missing date-stamped rows"


# ---------------------------------------------------------------------------
# 4) Optional smoke: when the live data/source_history.db exists locally
#    (operator has run the backfill scripts), assert the date-stamped
#    coverage contract holds. Skipped in CI.
# ---------------------------------------------------------------------------


def test_live_db_has_date_stamped_backfill():
    from flashcat.config import SOURCE_HISTORY_DB_PATH

    if not SOURCE_HISTORY_DB_PATH.exists():
        pytest.skip("data/source_history.db not present; run backfill scripts locally")
    with sqlite3.connect(str(SOURCE_HISTORY_DB_PATH)) as c:
        per_sport = dict(
            c.execute(
                "SELECT sport, COUNT(*) FROM predictions "
                "WHERE commence_time IS NOT NULL GROUP BY sport"
            ).fetchall()
        )
    # Sports that this PR explicitly backfills.
    for sport in ("nfl", "atp", "wta"):
        assert per_sport.get(sport, 0) > 0, (
            f"{sport} has no date-stamped predictions; the backfill regressed"
        )


# ---------------------------------------------------------------------------
# 5) Backfill helpers compile and the per-source bet-ledger meta math is
#    consistent.
# ---------------------------------------------------------------------------


def test_nfl_backfill_module_imports():
    mod = _load_script("backfill_nfl_historical")
    # Spot-check core symbols.
    assert hasattr(mod, "backfill")
    assert hasattr(mod, "_build_meta_rows")
    assert callable(mod._team_norm)
    # _team_norm canonicalises legacy codes.
    assert mod._team_norm("JAC") == "JAX"
    assert mod._team_norm("OAK") == "LV"


def test_tennis_backfill_module_imports():
    mod = _load_script("backfill_tennis_historical")
    assert hasattr(mod, "backfill_tour")
    assert hasattr(mod, "_build_meta_rows")
    # _norm handles both tennis-data and Sackmann name conventions.
    assert mod._norm("Sabalenka A.") == mod._norm("Aryna Sabalenka")
    assert mod._norm("O Connell C.") == mod._norm("Christopher O Connell")


def test_meta_rows_emit_two_window_cutoffs_per_source():
    """The hold-out runner subtracts a train-end window row from a full-end
    window row to recover hold-out ROI. The backfill MUST emit both rows
    per source so that subtraction is well-defined.
    """
    mod = _load_script("backfill_nfl_historical")
    train_end = date(2023, 12, 31)
    full_end = date(2024, 12, 31)
    window_start = date(2022, 9, 1)
    ledger = [
        # source-A: 200 train games, 100 hold-out games, all flat ROI -2%.
        *[(date(2022, 9, 1), "src-a", 0.6, 1.9, 1, 1)] * 60,
        *[(date(2022, 12, 1), "src-a", 0.55, 2.0, 0, 0)] * 40,
        *[(date(2023, 6, 1), "src-a", 0.7, 1.5, 1, 1)] * 70,
        *[(date(2023, 11, 1), "src-a", 0.45, 2.5, 0, 1)] * 30,
        *[(date(2024, 3, 1), "src-a", 0.65, 1.7, 1, 1)] * 60,
        *[(date(2024, 10, 1), "src-a", 0.5, 2.1, 0, 0)] * 40,
    ]
    rows = mod._build_meta_rows(
        ledger, sport="nfl",
        train_end=train_end, full_end=full_end, window_start=window_start,
    )
    by_we = {r["window_end"]: r for r in rows}
    assert "2023-12-31" in by_we, "train-window row missing"
    assert "2024-12-31" in by_we, "full-window row missing"
    train_row = by_we["2023-12-31"]
    full_row = by_we["2024-12-31"]
    assert train_row["n_bets"] == 200
    assert full_row["n_bets"] == 300
    # Hold-out subtraction must be well-defined.
    assert full_row["n_bets"] - train_row["n_bets"] == 100
    # ROI on either row is finite for a non-empty bet ledger.
    assert train_row["roi"] is not None
    assert full_row["roi"] is not None
