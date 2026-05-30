"""Regression tests for the blender de-dilution PR (PR-19).

Covers:
- Hard exclusion of sources whose ROI < ``blender_roi_floor`` (default
  -1%) when they have at least ``blender_min_bets_for_exclusion`` graded
  bets (default 50).
- ``min_sources_floor_active`` fallback: if applying the floor would
  leave fewer than 2 surviving sources, the exclusion is suppressed and
  a synthetic excluded-list entry is emitted.
- Low-sample cap: sources below ``min_bets_for_exclusion`` retain their
  softmax weight but are capped at 1/N of the pool.
- β=16 default (sharper concentration vs the legacy β=8).
- ``backtest_flat_stake`` headline simulator: flat $100, gated by the
  +3pp edge threshold, computes per-(sport, source) and blended ROI.
- ``scoreboard_patch.patch_scoreboard``: fills missing n_bets and
  synthesizes blended.roi when per-source meta has rows but the
  in-memory backtest produced ``None``.
- ``run_holdout_validation``: walk-forward 2022-2023 → 2024 split that
  refuses to silently overfit \u2014 large train-to-holdout degradation is
  surfaced rather than hidden.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from flashcat.backtest.flat_stake import (
    DEFAULT_EDGE_THRESHOLD,
    FLAT_STAKE,
    _meta_based_payload,
    _simulate_source,
    run_flat_stake_backtest,
)
from flashcat.backtest.scoreboard_patch import patch_scoreboard
from flashcat.config import (
    blender_min_bets_for_exclusion,
    blender_roi_floor,
    hybrid_beta,
)
from flashcat.model import reweight as rw


# ---------------------------------------------------------------------------
# 1. Hard exclusion floor
# ---------------------------------------------------------------------------

def test_blender_excludes_source_below_roi_floor(monkeypatch):
    """A source with ROI < floor AND n_bets >= 50 is hard-excluded."""
    monkeypatch.setattr(rw, "_overlay_source_history_for_reweight", lambda x: x)
    rows = {
        "src-good-a": {"n_events": 200, "n_bets": 200, "brier": 0.21, "roi": 0.04},
        "src-good-b": {"n_events": 200, "n_bets": 200, "brier": 0.21, "roi": 0.02},
        "src-bad":    {"n_events": 200, "n_bets": 200, "brier": 0.22, "roi": -0.05},
    }
    weights, excluded = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=16, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    assert "src-bad" not in weights
    excluded_names = {e["source"] for e in excluded if e.get("source")}
    assert "src-bad" in excluded_names
    # Reason explains the floor.
    bad = [e for e in excluded if e.get("source") == "src-bad"][0]
    assert "below floor" in bad["reason"]


def test_blender_keeps_low_sample_source(monkeypatch):
    """A source with ROI < floor BUT n_bets < min_bets_for_exclusion stays in."""
    monkeypatch.setattr(rw, "_overlay_source_history_for_reweight", lambda x: x)
    rows = {
        "src-good-a": {"n_events": 200, "n_bets": 200, "brier": 0.21, "roi": 0.04},
        "src-good-b": {"n_events": 200, "n_bets": 200, "brier": 0.21, "roi": 0.02},
        "src-low-n":  {"n_events": 100, "n_bets": 20,  "brier": 0.22, "roi": -0.10},
    }
    weights, excluded = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=16, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    # Low-n source stayed (didn't qualify for ROI exclusion).
    assert "src-low-n" in weights
    excluded_names = {e["source"] for e in excluded if e.get("source")}
    assert "src-low-n" not in excluded_names
    # And it got capped at 1/N (= 1/3 \u2248 0.333).
    assert weights["src-low-n"] <= (1.0 / 3) + 1e-9


def test_min_sources_floor_active_fallback(monkeypatch):
    """If ROI-floor exclusion would drop us below 2 sources, keep everyone."""
    monkeypatch.setattr(rw, "_overlay_source_history_for_reweight", lambda x: x)
    rows = {
        "src-bad-1": {"n_events": 200, "n_bets": 200, "brier": 0.22, "roi": -0.05},
        "src-bad-2": {"n_events": 200, "n_bets": 200, "brier": 0.23, "roi": -0.04},
    }
    weights, excluded = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=16, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    # Both sources kept (would have left zero survivors).
    assert "src-bad-1" in weights and "src-bad-2" in weights
    # Excluded list carries the fallback marker.
    fallback = [e for e in excluded if e.get("reason") == "min_sources_floor_active"]
    assert fallback, "expected min_sources_floor_active marker"
    assert set(fallback[0]["would_exclude"]) == {"src-bad-1", "src-bad-2"}


def test_min_sources_floor_active_keeps_at_least_two(monkeypatch):
    """One survivor is not enough \u2014 fallback keeps the would-excluded ones."""
    monkeypatch.setattr(rw, "_overlay_source_history_for_reweight", lambda x: x)
    rows = {
        "good":  {"n_events": 200, "n_bets": 200, "brier": 0.20, "roi": 0.05},
        "bad-1": {"n_events": 200, "n_bets": 200, "brier": 0.22, "roi": -0.05},
        "bad-2": {"n_events": 200, "n_bets": 200, "brier": 0.23, "roi": -0.04},
    }
    # With only one surviving good source, fallback kicks in.
    weights_strict, ex_strict = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=16, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    # Now there ARE >=2 survivors (good + something) only if we don't drop
    # both bads. With 1 good and 2 bads, dropping both bads leaves 1 \u2192
    # fallback engaged \u2192 bads stay.
    assert "bad-1" in weights_strict and "bad-2" in weights_strict
    fallback = [e for e in ex_strict if e.get("reason") == "min_sources_floor_active"]
    assert fallback


# ---------------------------------------------------------------------------
# 2. Sharper-\u03b2 default
# ---------------------------------------------------------------------------

def test_hybrid_beta_default_is_eight(monkeypatch):
    """PR #21 reverts β from the proposed 16 back to 8.

    PR #19 tried β=16 to concentrate weight on the top-Brier source, but
    the multi-sport hold-out evidence (PR #20) showed the higher β was
    overfitting. PR #21 ships β=8 — the env-configurable mechanism stays
    in place so operators can re-tune once a sport demonstrates real
    out-of-sample edge.
    """
    monkeypatch.delenv("FLASHCAT_HYBRID_BETA", raising=False)
    monkeypatch.delenv("FLASHCAT_BLENDER_BETA", raising=False)
    assert hybrid_beta() == 8.0


def test_blender_beta_env_var_overrides_legacy(monkeypatch):
    monkeypatch.setenv("FLASHCAT_BLENDER_BETA", "32")
    monkeypatch.setenv("FLASHCAT_HYBRID_BETA", "4")  # ignored
    assert hybrid_beta() == 32.0


def test_blender_roi_floor_default_disables_exclusion(monkeypatch):
    """PR #21 ships the floor at −1.0 (≡ −100% ROI), effectively disabled.

    The infrastructure (env-configurable floor + min-sources fallback +
    excluded-list reasons) stays in place so operators can re-enable
    exclusion via ``FLASHCAT_BLENDER_ROI_FLOOR`` once a sport has
    demonstrated positive hold-out edge — but the default does not
    exclude anything.
    """
    monkeypatch.delenv("FLASHCAT_BLENDER_ROI_FLOOR", raising=False)
    assert blender_roi_floor() == pytest.approx(-1.0)


def test_blender_min_bets_for_exclusion_default(monkeypatch):
    monkeypatch.delenv("FLASHCAT_BLENDER_MIN_BETS_FOR_EXCLUSION", raising=False)
    assert blender_min_bets_for_exclusion() == 50


# ---------------------------------------------------------------------------
# 3. Sharper-\u03b2 concentration vs legacy
# ---------------------------------------------------------------------------

def test_beta_16_concentrates_more_than_beta_8(monkeypatch):
    """\u03b2=16 should give the top-Brier source a larger share than \u03b2=8."""
    monkeypatch.setattr(rw, "_overlay_source_history_for_reweight", lambda x: x)
    rows = {
        "src-top":  {"n_events": 500, "n_bets": 500, "brier": 0.18, "roi": 0.06},
        "src-mid":  {"n_events": 500, "n_bets": 500, "brier": 0.22, "roi": 0.02},
        "src-low":  {"n_events": 500, "n_bets": 500, "brier": 0.24, "roi": 0.00},
    }
    w8, _ = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=8.0, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    w16, _ = rw._compute_pool_weights(
        rows, mode="brier_roi_hybrid", beta=16.0, lam=0.5, min_n=50,
        roi_floor=-0.01, min_bets_for_exclusion=50,
    )
    assert w16["src-top"] > w8["src-top"], (
        f"\u03b2=16 should put more weight on the top source than \u03b2=8 "
        f"\u2014 got w16={w16['src-top']:.3f} vs w8={w8['src-top']:.3f}"
    )


# ---------------------------------------------------------------------------
# 4. Flat $100 simulator
# ---------------------------------------------------------------------------

def test_flat_stake_simulator_uses_100_per_bet():
    """Every gated bet costs exactly $100."""
    # Synthetic rows: source picks home @ 60%, market close 1.91 (53% implied),
    # outcome=home wins. Edge = 7pp \u2192 above threshold.
    rows = [
        {"sport": "nfl", "source": "synthetic", "home_prob": 0.60,
         "home_won": 1, "market_close_decimal": 1.91},
        # Same setup, outcome=away.
        {"sport": "nfl", "source": "synthetic", "home_prob": 0.60,
         "home_won": 0, "market_close_decimal": 1.91},
        # No edge \u2014 skipped.
        {"sport": "nfl", "source": "synthetic", "home_prob": 0.53,
         "home_won": 1, "market_close_decimal": 1.91},
    ]
    sim = _simulate_source(rows, edge_threshold=DEFAULT_EDGE_THRESHOLD)
    assert sim.n_bets == 2
    assert sim.stake == 200.0  # 2 \u00d7 $100
    # Win $91, lose $100 \u2192 -$9, ROI -4.5%.
    assert sim.profit == pytest.approx(-9.0, rel=1e-3)
    assert sim.roi == pytest.approx(-0.045, rel=1e-3)


def test_flat_stake_simulator_skips_no_edge_rows():
    """Rows where (pick_prob - implied) < edge_threshold are not bet."""
    rows = [
        {"sport": "nfl", "source": "x", "home_prob": 0.51,
         "home_won": 1, "market_close_decimal": 2.00},  # 0.01 edge \u2014 too small
    ]
    sim = _simulate_source(rows, edge_threshold=0.03)
    assert sim.n_bets == 0
    assert sim.stake == 0.0


def test_flat_stake_meta_fallback(tmp_path):
    """When predictions lack market_close_decimal, the meta fallback fires."""
    db = tmp_path / "sh.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE predictions (
                event_id TEXT, sport TEXT, source TEXT,
                commence_time TEXT, home TEXT, away TEXT,
                home_prob REAL, home_won INTEGER,
                market_close_home REAL, market_close_decimal REAL,
                PRIMARY KEY (event_id, source)
            );
            CREATE TABLE meta (
                sport TEXT, source TEXT,
                window_start TEXT, window_end TEXT,
                n_events INTEGER, n_bets INTEGER, brier REAL,
                log_loss REAL, accuracy REAL, roi REAL,
                calibration_slope REAL, avg_clv_pp REAL,
                PRIMARY KEY (sport, source, window_end)
            );
        """)
        # No graded predictions, but a meta row claiming +5% ROI on 1,000 bets.
        conn.execute(
            "INSERT INTO meta VALUES ('mlb', 'good-src', '2022-01-01', "
            "'2024-12-31', 1000, 1000, 0.24, NULL, NULL, 0.05, NULL, NULL)"
        )

    payload = _meta_based_payload(
        db, stake=FLAT_STAKE, edge_threshold=0.03, weights=None,
    )
    assert payload
    assert payload["source"] == "meta"
    block = payload["per_sport"]["mlb"]["sources"]["good-src"]
    assert block["n_bets"] == 1000
    assert block["stake"] == pytest.approx(100_000)
    assert block["profit"] == pytest.approx(5_000)
    assert block["roi"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 5. Scoreboard patcher (addendum item 10)
# ---------------------------------------------------------------------------

def test_patch_scoreboard_fills_missing_n_bets_from_meta(tmp_path):
    """per_sport.sources[s].n_bets=None gets filled from meta.n_bets."""
    db = tmp_path / "sh.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE meta (
                sport TEXT, source TEXT,
                window_start TEXT, window_end TEXT,
                n_events INTEGER, n_bets INTEGER, brier REAL,
                log_loss REAL, accuracy REAL, roi REAL,
                calibration_slope REAL, avg_clv_pp REAL,
                PRIMARY KEY (sport, source, window_end)
            );
        """)
        conn.execute(
            "INSERT INTO meta VALUES ('mlb', 'srcA', '2022-01-01', "
            "'2024-12-31', 3000, 2400, 0.24, NULL, NULL, 0.06, NULL, NULL)"
        )

    sb_path = tmp_path / "sb.json"
    sb_path.write_text(json.dumps({
        "per_sport": {
            "mlb": {
                "n_events": 3000,
                "sources": {
                    "srcA": {"n_events": 3000, "brier": 0.24, "roi": None,
                              "n_bets": None, "wins": 0, "losses": 0},
                },
                "blended": None,
            }
        }
    }))
    patched = patch_scoreboard(sb_path, db, weights={"schema": "v2", "global": {},
                                                       "by_sport": {"mlb": {"srcA": 1.0}}})
    assert patched
    srcA = patched["per_sport"]["mlb"]["sources"]["srcA"]
    assert srcA["n_bets"] == 2400
    assert srcA["roi"] == pytest.approx(0.06)
    # Blended ROI synthesized.
    blended = patched["per_sport"]["mlb"]["blended"]
    assert blended is not None
    assert blended["roi"] == pytest.approx(0.06)
    assert blended["roi_source"] == "weighted_per_source_meta"


def test_patch_scoreboard_does_not_overwrite_real_blended(tmp_path):
    """If the backtest produced a real blended.roi, the patch must NOT touch it."""
    db = tmp_path / "sh.db"
    sqlite3.connect(str(db)).execute("CREATE TABLE meta (sport TEXT, source TEXT, "
                                       "window_start TEXT, window_end TEXT, "
                                       "n_events INTEGER, n_bets INTEGER, brier REAL, "
                                       "log_loss REAL, accuracy REAL, roi REAL, "
                                       "calibration_slope REAL, avg_clv_pp REAL, "
                                       "PRIMARY KEY (sport, source, window_end))").close()
    sb_path = tmp_path / "sb.json"
    real_blended = {"n_events": 540, "n_bets": 540, "roi": 0.13, "brier": 0.21}
    sb_path.write_text(json.dumps({
        "per_sport": {"nfl": {"n_events": 540, "sources": {}, "blended": real_blended}}
    }))
    patched = patch_scoreboard(sb_path, db, weights={"schema": "v2", "global": {},
                                                       "by_sport": {}})
    assert patched["per_sport"]["nfl"]["blended"]["roi"] == pytest.approx(0.13)
    assert patched["per_sport"]["nfl"]["blended"].get("roi_source") != "weighted_per_source_meta"


# ---------------------------------------------------------------------------
# 6. Walk-forward holdout (addendum item: PR's central de-risking deliverable)
# ---------------------------------------------------------------------------

def test_holdout_validation_returns_per_sport_table(tmp_path):
    """Holdout runner produces a per-sport result with train + holdout ROI."""
    from datetime import date

    from flashcat.model.holdout import run_holdout_validation

    db = tmp_path / "sh.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE predictions (
                event_id TEXT, sport TEXT, source TEXT,
                commence_time TEXT, home TEXT, away TEXT,
                home_prob REAL, home_won INTEGER,
                market_close_home REAL, market_close_decimal REAL,
                PRIMARY KEY (event_id, source)
            );
            CREATE TABLE meta (
                sport TEXT, source TEXT,
                window_start TEXT, window_end TEXT,
                n_events INTEGER, n_bets INTEGER, brier REAL,
                log_loss REAL, accuracy REAL, roi REAL,
                calibration_slope REAL, avg_clv_pp REAL,
                PRIMARY KEY (sport, source, window_end)
            );
        """)
        # Meta with training (2023-12-31) and full (2024-12-31) windows.
        # Training: +5% ROI on 1000 bets. Full: +3% ROI on 1500 bets.
        # \u2192 Holdout (2024 only): (3*1500 - 5*1000) / 500 = -1% ROI.
        conn.executemany(
            "INSERT INTO meta VALUES (?,?,?,?,?,?,?,NULL,NULL,?,NULL,NULL)",
            [
                ("mlb", "src1", "2022-01-01", "2023-12-31", 1000, 1000, 0.24, 0.05),
                ("mlb", "src1", "2022-01-01", "2024-12-31", 1500, 1500, 0.24, 0.03),
                ("mlb", "src2", "2022-01-01", "2023-12-31", 1000, 1000, 0.25, 0.02),
                ("mlb", "src2", "2022-01-01", "2024-12-31", 1500, 1500, 0.25, 0.01),
            ],
        )

    results = run_holdout_validation(
        db, beta=16, lam=0.5, roi_floor=-0.01, min_bets_for_exclusion=50,
        mode="brier_roi_hybrid",
        train_start=date(2022, 1, 1), train_end=date(2023, 12, 31),
        holdout_start=date(2024, 1, 1), holdout_end=date(2024, 12, 31),
    )
    assert "mlb" in results
    r = results["mlb"]
    # Both sources made it into the blend (none below floor).
    assert set(r.sources_in_blend) == {"src1", "src2"}
    # Training ROI is the weighted average of source ROIs in training.
    assert r.train_roi is not None
    # Holdout ROI is reconstructed from full-train and should be negative.
    assert r.holdout_roi is not None
    assert r.holdout_roi < r.train_roi, (
        "expected hold-out ROI to be lower than training when full-window "
        "ROI is below training-window ROI"
    )


def test_holdout_validation_handles_empty_db(tmp_path):
    """Empty source_history.db must not raise."""
    from flashcat.model.holdout import run_holdout_validation
    db = tmp_path / "sh.db"
    # Create with no tables.
    sqlite3.connect(str(db)).close()
    results = run_holdout_validation(db_path=db)
    assert results == {}


def test_holdout_validation_no_db(tmp_path):
    from flashcat.model.holdout import run_holdout_validation
    results = run_holdout_validation(db_path=tmp_path / "does_not_exist.db")
    assert results == {}


# ---------------------------------------------------------------------------
# 7. The de-risking gate: hold-out ROI cannot be catastrophically lower
# than training ROI on a sport with enough hold-out bets.
# ---------------------------------------------------------------------------

def test_holdout_validation_flags_overfit_when_holdout_collapses(tmp_path):
    """If training ROI >> hold-out ROI on a sport with >= 200 hold-out bets,
    the result carries a negative ``delta_pp`` and the PR write-up MUST
    surface it (rather than declaring victory on the training number).
    """
    from datetime import date

    from flashcat.model.holdout import run_holdout_validation

    db = tmp_path / "sh.db"
    with sqlite3.connect(str(db)) as conn:
        conn.executescript("""
            CREATE TABLE predictions (
                event_id TEXT, sport TEXT, source TEXT,
                commence_time TEXT, home TEXT, away TEXT,
                home_prob REAL, home_won INTEGER,
                market_close_home REAL, market_close_decimal REAL,
                PRIMARY KEY (event_id, source)
            );
            CREATE TABLE meta (
                sport TEXT, source TEXT,
                window_start TEXT, window_end TEXT,
                n_events INTEGER, n_bets INTEGER, brier REAL,
                log_loss REAL, accuracy REAL, roi REAL,
                calibration_slope REAL, avg_clv_pp REAL,
                PRIMARY KEY (sport, source, window_end)
            );
        """)
        # Train: +8% ROI on 2000 bets. Full: +1% ROI on 3000 bets.
        # Holdout reconstruction: (1*3000 - 8*2000)/1000 = -13% ROI on 1000 bets.
        conn.executemany(
            "INSERT INTO meta VALUES (?,?,?,?,?,?,?,NULL,NULL,?,NULL,NULL)",
            [
                ("mlb", "s1", "2022-01-01", "2023-12-31", 2000, 2000, 0.24, 0.08),
                ("mlb", "s1", "2022-01-01", "2024-12-31", 3000, 3000, 0.24, 0.01),
                ("mlb", "s2", "2022-01-01", "2023-12-31", 2000, 2000, 0.25, 0.06),
                ("mlb", "s2", "2022-01-01", "2024-12-31", 3000, 3000, 0.25, 0.00),
            ],
        )

    results = run_holdout_validation(
        db, beta=16, lam=0.5, roi_floor=-0.01, min_bets_for_exclusion=50,
        mode="brier_roi_hybrid",
        train_start=date(2022, 1, 1), train_end=date(2023, 12, 31),
        holdout_start=date(2024, 1, 1), holdout_end=date(2024, 12, 31),
    )
    r = results["mlb"]
    assert r.train_roi is not None and r.holdout_roi is not None
    # Specifically: holdout is much worse than train \u2014 the PR write-up should
    # flag this rather than report the training number as the headline.
    assert r.delta_pp is not None
    assert r.delta_pp < -5.0, (
        f"expected hold-out to collapse vs training by >5pp; "
        f"got delta_pp={r.delta_pp:.2f}"
    )
    assert r.holdout_n_bets >= 200, (
        "synthetic holdout window should have enough bets for the gate"
    )


# ---------------------------------------------------------------------------
# 8. PR #21 — fix the `_blended_roi` two-way-market settlement bug.
# ---------------------------------------------------------------------------

def test_blended_roi_settles_balanced_two_way_at_vig():
    """Phil's regression case (PR #21).

    Synthesizes a balanced two-way moneyline market (home -110 / away -110,
    both 1.91 decimal). The model predicts 50% home_prob and the synthetic
    "source" picks the higher-prob side (home, by the >=0.5 tie-break)
    storing 1.91 for the picked side. Across 10 events with 5 home wins +
    5 away wins, every event is a "pick home" → 5 wins × $91 profit, 5
    losses × $100 stake-lost.

    Expected: ROI ≈ -4.5% (the vig). NOT zero (no edge, the vig is real).
    NOT +35% (the original PR #19 bug routed by `_meta_based_payload` to
    work around it).
    """
    from flashcat.model.holdout import _blended_roi

    rows: list[dict] = []
    # 5 events where home wins; 5 where away wins. Source always picks
    # home (home_prob=0.50, >=0.5 → picks home), stores 1.91 on the home
    # side. We DON'T populate market_close_home because the picked side
    # is observable directly.
    for i in range(5):
        rows.append({
            "event_id": f"home-wins-{i}",
            "source": "synthetic",
            "home_prob": 0.50,
            "home_won": 1,
            "market_close_decimal": 1.91,
        })
    for i in range(5):
        rows.append({
            "event_id": f"away-wins-{i}",
            "source": "synthetic",
            "home_prob": 0.50,
            "home_won": 0,
            "market_close_decimal": 1.91,
        })
    weights = {"synthetic": 1.0}
    roi, n_bets, n_wins = _blended_roi(rows, weights)
    assert n_bets == 10
    assert n_wins == 5
    # 5 wins × $91 + 5 losses × −$100 = −$45 on $1000 stake = −4.5%.
    # Spec rounds this to ≈ −4.55% (matches what a real −110/−110 pair
    # vig works out to with 1.909 decimal); we accept either rounding.
    assert roi is not None
    assert -0.05 < roi < -0.04, (
        f"expected ROI near the vig (≈ −4.5%), got {roi*100:+.2f}%"
    )


def test_blended_roi_uses_picked_side_decimal_not_first_row():
    """The original PR #19 bug: with two sources picking opposite sides,
    the function grabbed the FIRST row's decimal and settled the blend
    at it — even when the blend's pick was the OPPOSITE side. The fix
    must settle at the picked side's own decimal.

    Construct an event where:
      * Source A picks HOME (home_prob=0.65), stores home_dec = 1.40.
      * Source B picks AWAY (home_prob=0.30), stores away_dec = 3.50.
      * Equal weights → blended home_prob = 0.475 → pick AWAY.
      * Outcome: away wins.

    Buggy behaviour (using first row's decimal): if source A's row is
    first, would settle AWAY pick at 1.40 → profit = $40 on a $100 stake.
    Fixed behaviour: settles AWAY pick at the away decimal = 3.50 →
    profit = $250 on a $100 stake.
    """
    from flashcat.model.holdout import _blended_roi

    rows = [
        {
            "event_id": "evt1",
            "source": "src-A",
            "home_prob": 0.65,
            "home_won": 0,  # away wins
            "market_close_decimal": 1.40,  # home decimal (A picked home)
        },
        {
            "event_id": "evt1",
            "source": "src-B",
            "home_prob": 0.30,
            "home_won": 0,
            "market_close_decimal": 3.50,  # away decimal (B picked away)
        },
    ]
    weights = {"src-A": 0.5, "src-B": 0.5}
    roi, n_bets, n_wins = _blended_roi(rows, weights)
    # Blend = 0.5*0.65 + 0.5*0.30 = 0.475 → pick away.
    # Away wins → we win, settled at the away decimal 3.50.
    # Profit = $250 on $100 stake → ROI = +2.50.
    assert n_bets == 1
    assert n_wins == 1
    assert roi == pytest.approx(2.50, rel=1e-3)


def test_blended_roi_derives_missing_side_from_market_close_home():
    """When only one side's decimal is observable in the rows, fall back to
    the no-vig inverse using ``market_close_home`` (the devigged home
    probability) to derive the other side.

    Setup: single source picks HOME (home_prob=0.70), stores
    home_dec=1.50, plus market_close_home=0.60 (devigged). The blend
    inherits 0.70 → picks home, settles at the home decimal directly
    (sanity check: ROI on a winning bet = home_dec - 1 = 0.50).
    """
    from flashcat.model.holdout import _blended_roi, _settlement_decimals

    rs = [
        {
            "event_id": "evt2",
            "source": "src-X",
            "home_prob": 0.70,
            "home_won": 1,
            "market_close_home": 0.60,
            "market_close_decimal": 1.50,
        },
    ]
    home_dec, away_dec, home_won = _settlement_decimals(rs)
    assert home_dec == pytest.approx(1.50)
    # Derived: away_dec = p_h * home_dec / (1 - p_h) = 0.6 * 1.5 / 0.4 = 2.25
    assert away_dec == pytest.approx(2.25)
    assert home_won is True

    # Now a flip: same event but source picked AWAY → stores away_dec.
    # Derived home_dec = (1 - p_h) * away_dec / p_h = 0.4 * 2.25 / 0.6 = 1.50.
    rs2 = [
        {
            "event_id": "evt3",
            "source": "src-Y",
            "home_prob": 0.20,
            "home_won": 1,
            "market_close_home": 0.60,
            "market_close_decimal": 2.25,
        },
    ]
    home_dec2, away_dec2, _ = _settlement_decimals(rs2)
    assert away_dec2 == pytest.approx(2.25)
    assert home_dec2 == pytest.approx(1.50)

    # End-to-end through _blended_roi on the first row only.
    weights = {"src-X": 1.0}
    roi, n_bets, n_wins = _blended_roi(rs, weights)
    assert n_bets == 1 and n_wins == 1
    assert roi == pytest.approx(0.50, rel=1e-3)
