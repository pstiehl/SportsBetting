"""Tests for ``flashcat.source_accountability``.

We test the math primitives, verdict bucketing, the tennis-data parser with
an injected fake fetcher, and the end-to-end ``assemble_report`` flow
against a tmp DB. No network, no real tennis-data pulls.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date

import pytest

from flashcat import source_accountability as sa
from flashcat import source_history as sh


# ---------------------------------------------------------------------------
# Math primitives
# ---------------------------------------------------------------------------


def test_brier_score_basic():
    # Perfect prediction → 0
    assert sa.brier_score(1.0, 1) == 0.0
    assert sa.brier_score(0.0, 0) == 0.0
    # Worst-possible prediction → 1
    assert sa.brier_score(1.0, 0) == 1.0
    assert sa.brier_score(0.0, 1) == 1.0
    # Coin-flip → 0.25
    assert sa.brier_score(0.5, 1) == 0.25
    assert sa.brier_score(0.5, 0) == 0.25


def test_log_loss_basic():
    # Confident-correct → near zero
    assert sa.log_loss(0.99, 1) < 0.02
    # Confident-wrong → blows up but clipped (won't be inf)
    assert sa.log_loss(0.999, 0) < 10.0
    # Coin-flip → ln(2)
    assert abs(sa.log_loss(0.5, 1) - math.log(2)) < 1e-9


# ---------------------------------------------------------------------------
# settle_ticks
# ---------------------------------------------------------------------------


def _tick(d: date, home_prob: float, home_won: int, picked_dec: float):
    """Helper: build a BetTick that picks the higher-prob side."""
    picked_home = home_prob >= 0.5
    won = (picked_home and home_won == 1) or (not picked_home and home_won == 0)
    profit = 100.0 * (picked_dec - 1.0) if won else -100.0
    return sa.BetTick(
        date=d, sport="test", source="t", home_prob=home_prob,
        home_won=home_won, picked_home=picked_home, picked_dec=picked_dec,
        won=won, profit_100=profit,
    )


def test_settle_ticks_empty():
    out = sa.settle_ticks([])
    assert out["n_predictions"] == 0
    assert out["hit_rate"] is None
    assert out["roi_flat_100"] is None


def test_settle_ticks_all_wins_at_even_money():
    # Three wins at decimal 2.00 → profit = +$300, ROI = +100%, no losing streak.
    ticks = [
        _tick(date(2024, 1, 1), 0.6, 1, 2.0),
        _tick(date(2024, 1, 2), 0.6, 1, 2.0),
        _tick(date(2024, 1, 3), 0.6, 1, 2.0),
    ]
    out = sa.settle_ticks(ticks)
    assert out["n_predictions"] == 3
    assert out["wins"] == 3
    assert out["losses"] == 0
    assert out["hit_rate"] == 1.0
    assert out["wagered_usd"] == 300.0
    assert out["profit_usd"] == pytest.approx(300.0)
    assert out["roi_flat_100"] == pytest.approx(1.0)
    assert out["longest_losing_streak"] == 0
    assert out["max_drawdown_usd"] == 0.0


def test_settle_ticks_losing_streak_and_drawdown():
    # Win $100, lose $100, lose $100, lose $100, win $100.
    # Bankroll curve: 100, 0, -100, -200, -100. Peak = 100, trough = -200,
    # max DD = 300. Longest losing streak = 3.
    ticks = [
        _tick(date(2024, 1, 1), 0.6, 1, 2.0),  # win +100
        _tick(date(2024, 1, 2), 0.6, 0, 2.0),  # loss -100
        _tick(date(2024, 1, 3), 0.6, 0, 2.0),  # loss -100
        _tick(date(2024, 1, 4), 0.6, 0, 2.0),  # loss -100
        _tick(date(2024, 1, 5), 0.6, 1, 2.0),  # win +100
    ]
    out = sa.settle_ticks(ticks)
    assert out["wins"] == 2
    assert out["losses"] == 3
    assert out["longest_losing_streak"] == 3
    assert out["max_drawdown_usd"] == 300.0


# ---------------------------------------------------------------------------
# Verdict bucketing
# ---------------------------------------------------------------------------


def test_verdict_insufficient_data():
    assert sa.verdict_for({"n_predictions": 10, "brier": 0.20, "roi_flat_100": 0.05}) == "INSUFFICIENT-DATA"


def test_verdict_drop_brier_above_coin_flip():
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.26, "roi_flat_100": 0.0}) == "DROP"


def test_verdict_drop_roi_below_minus_ten():
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.22, "roi_flat_100": -0.15}) == "DROP"


def test_verdict_keep_with_caveats_near_breakeven():
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.22, "roi_flat_100": -0.01}) == "KEEP-WITH-CAVEATS"
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.22, "roi_flat_100": +0.005}) == "KEEP-WITH-CAVEATS"


def test_verdict_keep_positive_roi():
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.20, "roi_flat_100": 0.05}) == "KEEP"


def test_verdict_noise():
    # Brier in vig territory (.24-.25), small negative ROI
    assert sa.verdict_for({"n_predictions": 1000, "brier": 0.245, "roi_flat_100": -0.03}) == "NOISE"


# ---------------------------------------------------------------------------
# Tennis-data parser with injected fetcher
# ---------------------------------------------------------------------------


def _fake_tennis_fetch(tour: str, year: int) -> list[dict]:
    """Return a tiny ATP fixture: two matches in 2022, both completed."""
    return [
        {
            "Date": date(2022, 1, 3), "Winner": "Alcaraz", "Loser": "Borges",
            "Comment": "Completed",
            "WPts": 1500, "LPts": 600,
            "PSW": 1.30, "PSL": 3.60,
            "B365W": 1.29, "B365L": 3.50,
            "AvgW": 1.32, "AvgL": 3.40,
        },
        {
            "Date": date(2022, 1, 4), "Winner": "Berretini", "Loser": "Alcaraz",
            "Comment": "Completed",
            "WPts": 1200, "LPts": 1500,
            "PSW": 2.10, "PSL": 1.75,
            "B365W": 2.10, "B365L": 1.75,
            "AvgW": 2.08, "AvgL": 1.78,
        },
    ]


def test_tennis_ticks_from_archive_uses_fetcher():
    rows = sa.tennis_ticks_from_archive("atp", years=[2022], fetcher=_fake_tennis_fetch)
    assert len(rows) == 2
    assert {r["winner"] for r in rows} == {"Alcaraz", "Berretini"}
    # Pinnacle preferred over Bet365 / Avg.
    assert rows[0]["w_dec"] == 1.30
    assert rows[0]["l_dec"] == 3.60


def test_build_tennis_per_source_ticks_emits_all_sources():
    ticks_by_src = sa.build_tennis_per_source_ticks(
        "atp", years=[2022], fetcher=_fake_tennis_fetch
    )
    # Every source should produce one tick per match.
    for src in ("market-close", "tennis-rank-bt", "coin-flip"):
        assert src in ticks_by_src
        assert len(ticks_by_src[src]) == 2
    # market-close is calibrated: home_prob ∈ (0,1).
    for t in ticks_by_src["market-close"]:
        assert 0.0 < t.home_prob < 1.0
    # coin-flip is exactly 0.5 always.
    for t in ticks_by_src["coin-flip"]:
        assert t.home_prob == 0.5


# ---------------------------------------------------------------------------
# DB metrics
# ---------------------------------------------------------------------------


def test_per_source_db_metrics_against_seeded_db(tmp_path, monkeypatch):
    db = tmp_path / "source_history.db"
    monkeypatch.setattr(sh, "SOURCE_HISTORY_DB_PATH", db)
    monkeypatch.setattr(sa, "SOURCE_HISTORY_DB_PATH", db)
    sh.init_db(db)
    sh.upsert_predictions(
        [
            {"event_id": "e1", "sport": "nfl", "source": "test-src",
             "commence_time": "2024-01-01T18:00:00+00:00",
             "home": "A", "away": "B", "home_prob": 0.7, "home_won": 1,
             "market_close_home": None, "market_close_decimal": None},
            {"event_id": "e2", "sport": "nfl", "source": "test-src",
             "commence_time": "2024-01-08T18:00:00+00:00",
             "home": "C", "away": "D", "home_prob": 0.6, "home_won": 0,
             "market_close_home": None, "market_close_decimal": None},
        ],
        path=db,
    )
    metrics = sa.per_source_db_metrics(db_path=db)
    key = ("nfl", "test-src")
    assert key in metrics
    m = metrics[key]
    assert m["n_predictions"] == 2
    assert m["wins"] == 1  # 0.7 picked home, won; 0.6 picked home, lost
    assert m["hit_rate"] == 0.5
    # Brier = ((0.7-1)^2 + (0.6-0)^2)/2 = (0.09 + 0.36)/2 = 0.225
    assert m["brier"] == pytest.approx(0.225, abs=1e-9)


# ---------------------------------------------------------------------------
# Predict.tennis scorecard — make sure provenance / structure is intact.
# ---------------------------------------------------------------------------


def test_predict_tennis_scorecard_shape():
    sc = sa.predict_tennis_scorecard()
    assert "atp" in sc and "wta" in sc
    assert sc["atp"]["n_predictions"] > 200  # has real data
    assert sc["wta"]["n_predictions"] > 200
    # Hit rates are >0.6 (the tour is predictable; the SITE says so)
    assert 0.5 < sc["atp"]["hit_rate"] < 1.0
    assert 0.5 < sc["wta"]["hit_rate"] < 1.0
    # Self-reported yield is negative (per the 2024 review)
    assert sc["overall_self_reported_yield"] < 0
    # Provenance is non-empty and mentions predict.tennis.
    assert "predict.tennis" in sc["provenance"]


# ---------------------------------------------------------------------------
# End-to-end assemble_report
# ---------------------------------------------------------------------------


def test_assemble_report_end_to_end(tmp_path, monkeypatch):
    db = tmp_path / "source_history.db"
    monkeypatch.setattr(sh, "SOURCE_HISTORY_DB_PATH", db)
    monkeypatch.setattr(sa, "SOURCE_HISTORY_DB_PATH", db)
    sh.init_db(db)
    sh.upsert_predictions(
        [
            {"event_id": f"e{i}", "sport": "nfl", "source": "calibrated-src",
             "commence_time": "2024-01-01T18:00:00+00:00",
             "home": "A", "away": "B",
             "home_prob": 0.7 if i % 2 else 0.3,
             "home_won": (i % 3 == 0),
             "market_close_home": None, "market_close_decimal": None}
            for i in range(300)
        ],
        path=db,
    )
    report = sa.assemble_report(
        db_path=db,
        tennis_years=[2022],
        include_tennis_per_event=True,
        tennis_fetcher=_fake_tennis_fetch,
    )
    assert report["n_sources"] >= 3  # nfl source + atp/wta sources + predict.tennis rows
    # Render check
    md = sa.render_markdown(report)
    assert "Source Accountability Report" in md
    assert "Verdict roll-up" in md
    assert "Honesty pact" in md
    assert "predict.tennis" in md


def test_assemble_report_skip_tennis(tmp_path, monkeypatch):
    db = tmp_path / "source_history.db"
    monkeypatch.setattr(sh, "SOURCE_HISTORY_DB_PATH", db)
    monkeypatch.setattr(sa, "SOURCE_HISTORY_DB_PATH", db)
    sh.init_db(db)
    report = sa.assemble_report(
        db_path=db,
        include_tennis_per_event=False,
    )
    # No per-event tennis sources → only the predict.tennis observed-external rows.
    md = sa.render_markdown(report)
    assert "predict.tennis" in md
    # Still well-formed
    assert "Verdict roll-up" in md


# ---------------------------------------------------------------------------
# Write-report side-effects (dated + latest mirror).
# ---------------------------------------------------------------------------


def test_write_report_creates_dated_and_latest(tmp_path):
    report = {
        "generated_at": "2026-05-31T00:00:00Z",
        "n_sources": 0,
        "sources": [],
    }
    dated, latest = sa.write_report(report, out_dir=tmp_path)
    assert dated.exists()
    assert latest.exists()
    # JSON sidecar
    assert dated.with_suffix(".json").exists()
    assert latest.with_suffix(".json").exists()
    # Latest mirrors dated
    assert dated.read_text() == latest.read_text()
