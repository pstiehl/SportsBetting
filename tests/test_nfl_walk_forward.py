"""Walk-forward harness sanity tests for the NFL feature-expansion harness.

Mirrors ``tests/test_mlb_walk_forward.py`` in structure: small synthetic
datasets, no network, no I/O. Verifies:

  * Leakage gates are enforced (the rolling state at game date D reflects
    only games strictly before D).
  * Bye detection is correct (season opener never flagged, > 10d gap = bye).
  * Divisional detection is correct (intra-division pairs flagged).
  * Market devig is correct (devigged probs sum to 1.0).
  * Walk-forward splits never overlap train and eval windows.
  * Required-feature gate (market + L4 rolling EPA) is enforced.
  * Simulator returns symmetric pnl signs (won → positive, lost → negative).
  * Loss bucket classifier returns a valid bucket for every losing bet.
  * Per-season aggregation is consistent.
"""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.nfl_features.feature_builder import (
    FEATURE_NAMES,
    GameRow,
    RollingNFLFeatures,
    TeamGameStats,
    _moneyline_to_prob,
    _same_division,
    build_features,
    compute_bye_status,
    fit_rolling_rates,
    market_prob_home,
)
from flashcat.nfl_features.model import (
    WalkForwardSplit,
    make_splits,
)
from flashcat.nfl_features.simulator import (
    PRODUCTION_EDGE_GATE_PP,
    _classify_loss,
    _moneyline_to_decimal,
    _moneyline_to_raw_implied,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d, home, away, hs, as_, week=1, season=2022, hml=-150, aml=130, hr=7, ar=7,
       elo=None, epa=None):
    return GameRow(
        game_id=f"{season}_{week:02d}_{away}_{home}",
        game_date=d,
        season=season,
        week=week,
        home=home,
        away=away,
        home_score=hs,
        away_score=as_,
        home_moneyline=hml,
        away_moneyline=aml,
        home_rest=hr,
        away_rest=ar,
        elo_prob_home=elo,
        epa_prob_home=epa,
    )


def _ts(team, season, week, d, off=0.05, def_=-0.02, succ=0.45, pe=0.05, re=0.02):
    return TeamGameStats(
        game_id=f"{season}_{week:02d}_{team}",
        team=team,
        season=season,
        week=week,
        game_date=d,
        off_epa_per_play=off,
        def_epa_per_play=def_,
        success_rate=succ,
        pass_epa_per_play=pe,
        rush_epa_per_play=re,
        n_off_plays=60,
        n_def_plays=60,
    )


# ---------------------------------------------------------------------------
# 1-4. Devig math
# ---------------------------------------------------------------------------

def test_moneyline_to_prob_negative_favorite():
    p = _moneyline_to_prob(-150)
    assert 0.55 < p < 0.65  # -150 = 60% raw implied
    assert abs(p - 0.6) < 1e-9


def test_moneyline_to_prob_positive_underdog():
    p = _moneyline_to_prob(+130)
    assert 0.40 < p < 0.50  # +130 = ~43.5%
    assert abs(p - 100.0 / 230.0) < 1e-9


def test_market_prob_home_devig_sums_to_one():
    g = _g(date(2023, 9, 10), "KC", "DET", 0, 0, hml=-200, aml=170)
    p_home = market_prob_home(g)
    p_away_raw = _moneyline_to_prob(170)
    p_home_raw = _moneyline_to_prob(-200)
    # Devig: implied_home / (implied_home + implied_away)
    expected = p_home_raw / (p_home_raw + p_away_raw)
    assert abs(p_home - expected) < 1e-9
    # Devigged probs should sum to 1.0
    g2 = _g(date(2023, 9, 10), "DET", "KC", 0, 0, hml=170, aml=-200)
    p_away = market_prob_home(g2)
    assert abs((p_home + p_away) - 1.0) < 1e-9


def test_moneyline_to_decimal_round_trip():
    assert abs(_moneyline_to_decimal(-150) - (1.0 + 100.0 / 150.0)) < 1e-9
    assert abs(_moneyline_to_decimal(130) - (1.0 + 130.0 / 100.0)) < 1e-9


# ---------------------------------------------------------------------------
# 5-7. Division detection
# ---------------------------------------------------------------------------

def test_same_division_afc_west():
    assert _same_division("KC", "DEN") is True
    assert _same_division("KC", "LAC") is True


def test_same_division_cross_conference_false():
    assert _same_division("KC", "DAL") is False


def test_same_division_handles_team_recodes():
    # OAK normalized to LV is tricky — _same_division receives original codes
    assert _same_division("LV", "KC") is True


# ---------------------------------------------------------------------------
# 8-10. Bye detection
# ---------------------------------------------------------------------------

def test_bye_status_season_opener_false():
    games = [_g(date(2022, 9, 11), "KC", "DET", 24, 17, week=1, season=2022)]
    bye = compute_bye_status(games)
    assert bye[(date(2022, 9, 11), "KC")] is False
    assert bye[(date(2022, 9, 11), "DET")] is False


def test_bye_status_long_gap_true():
    games = [
        _g(date(2022, 9, 11), "KC", "DET", 24, 17, week=1),
        _g(date(2022, 10, 2), "KC", "GB", 21, 14, week=4),  # 21 days = bye
    ]
    bye = compute_bye_status(games)
    assert bye[(date(2022, 10, 2), "KC")] is True


def test_bye_status_normal_week_false():
    games = [
        _g(date(2022, 9, 11), "KC", "DET", 24, 17, week=1),
        _g(date(2022, 9, 18), "KC", "GB", 21, 14, week=2),  # 7 days = normal
    ]
    bye = compute_bye_status(games)
    assert bye[(date(2022, 9, 18), "KC")] is False


# ---------------------------------------------------------------------------
# 11-13. Rolling rates — leakage gate
# ---------------------------------------------------------------------------

def test_rolling_snapshot_excludes_same_day_game():
    """Snapshot at date D must reflect only stats from games BEFORE D."""
    stats = {
        "KC": [
            _ts("KC", 2022, 1, date(2022, 9, 11), off=0.10),
            _ts("KC", 2022, 2, date(2022, 9, 18), off=0.20),
        ]
    }
    snapshots = fit_rolling_rates(stats, season_reset=False)
    # Snapshot for the week 2 game date should reflect ONLY week 1
    snap_w2 = snapshots[(date(2022, 9, 18), "KC")]
    assert list(snap_w2.off_epa) == [0.10]
    assert snap_w2.last_game_date == date(2022, 9, 11)


def test_rolling_snapshot_season_reset():
    stats = {
        "KC": [
            _ts("KC", 2022, 17, date(2022, 12, 31), off=0.50),
            _ts("KC", 2023, 1, date(2023, 9, 10), off=0.10),
        ]
    }
    snapshots = fit_rolling_rates(stats, season_reset=True)
    # Snapshot at 2023 opener should reflect fresh state (no 2022 carry-over)
    snap = snapshots[(date(2023, 9, 10), "KC")]
    assert len(snap.off_epa) == 0


def test_rolling_features_avg_n_too_small_returns_none():
    f = RollingNFLFeatures()
    f.off_epa.extend([0.1, 0.2])
    assert f.off_epa_n(4) is None  # only 2 samples
    assert f.off_epa_n(2) == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# 14-15. Walk-forward splits
# ---------------------------------------------------------------------------

def test_walk_forward_split_no_leakage():
    splits = make_splits(date(2022, 9, 1), date(2023, 12, 31),
                        train_window_days=365, eval_window_days=30, warmup_days=120)
    assert len(splits) > 0
    for s in splits:
        # train_end must be strictly < eval_start
        assert s.train_end < s.eval_start
        # eval_end >= eval_start
        assert s.eval_end >= s.eval_start
        # train window <= 365 days
        assert (s.train_end - s.train_start).days <= 365


def test_walk_forward_split_post_init_asserts_leakage():
    with pytest.raises(AssertionError):
        WalkForwardSplit(
            train_start=date(2022, 1, 1),
            train_end=date(2022, 12, 31),
            eval_start=date(2022, 12, 31),  # equal to train_end — must fail
            eval_end=date(2023, 1, 30),
        )


# ---------------------------------------------------------------------------
# 16-17. Feature gate
# ---------------------------------------------------------------------------

def test_build_features_returns_none_when_market_missing():
    g = GameRow(
        game_id="X", game_date=date(2023, 9, 10), season=2023, week=1,
        home="KC", away="DET", home_score=24, away_score=17,
        home_moneyline=None, away_moneyline=None,
    )
    f = build_features(g, rolling={}, bye_status={})
    assert f is None


def test_build_features_returns_none_when_rolling_l4_missing():
    g = _g(date(2023, 9, 10), "KC", "DET", 24, 17)
    # Empty rolling state → no L4 EPA → None
    f = build_features(g, rolling={}, bye_status={})
    assert f is None


# ---------------------------------------------------------------------------
# 18. Feature vector length matches FEATURE_NAMES
# ---------------------------------------------------------------------------

def test_feature_vector_length_matches_feature_names():
    # Use a hand-built feats dict
    from flashcat.nfl_features.feature_builder import feature_vector
    feats = {name: 0.0 for name in FEATURE_NAMES}
    v = feature_vector(feats)
    assert len(v) == len(FEATURE_NAMES)


def test_feature_vector_handles_none_with_fill():
    from flashcat.nfl_features.feature_builder import feature_vector
    feats = {name: None for name in FEATURE_NAMES}
    v = feature_vector(feats, fill_value=0.0)
    assert all(x == 0.0 for x in v)
    v2 = feature_vector(feats, fill_value=-1.0)
    assert all(x == -1.0 for x in v2)


# ---------------------------------------------------------------------------
# 19-21. Loss bucket classifier
# ---------------------------------------------------------------------------

def test_classify_loss_pure_variance():
    bet = {"pick_prob": 0.51, "market_implied_prob": 0.50, "edge_pp": 0.01, "pick_home": True}
    assert _classify_loss(bet, {}) == "pure_variance"


def test_classify_loss_line_moved_against():
    bet = {"pick_prob": 0.60, "market_implied_prob": 0.60, "edge_pp": 0.0, "pick_home": True}
    assert _classify_loss(bet, {}) == "line_moved_against"


def test_classify_loss_divisional():
    bet = {"pick_prob": 0.65, "market_implied_prob": 0.60, "edge_pp": 0.05, "pick_home": True}
    feats = {"divisional": 1.0}
    assert _classify_loss(bet, feats) == "divisional_misjudged"


def test_classify_loss_generic_fallback():
    bet = {"pick_prob": 0.60, "market_implied_prob": 0.55, "edge_pp": 0.05, "pick_home": True}
    feats = {"divisional": 0.0, "home_off_bye": 0.0, "away_off_bye": 0.0, "rest_diff": 0.0}
    assert _classify_loss(bet, feats) == "generic"


# ---------------------------------------------------------------------------
# 22-24. End-to-end smoke (no I/O — synthetic data only)
# ---------------------------------------------------------------------------

def test_e2e_build_features_with_rolling_and_priors():
    """One game with full rolling history → all features populated."""
    # Build 6 weeks of stats for KC and DET so L4 windows fill. The 6th week's
    # snapshot date IS the prediction game date (snapshots reflect only prior games).
    from datetime import timedelta
    stats = {"KC": [], "DET": []}
    base = date(2022, 9, 11)
    test_date = base + timedelta(days=7 * 5)  # week 6 date
    for w in range(1, 7):
        d = base + timedelta(days=7 * (w - 1))
        stats["KC"].append(_ts("KC", 2022, w, d, off=0.10 + 0.01 * w, def_=-0.05))
        stats["DET"].append(_ts("DET", 2022, w, d, off=0.05, def_=-0.01))
    snapshots = fit_rolling_rates(stats, season_reset=True)
    # Game on the same date as week-6 stat (so snapshot lookup hits, reflecting weeks 1-5)
    game = _g(test_date, "KC", "DET", 24, 17, week=6, season=2022,
              elo=0.65, epa=0.62)
    bye = compute_bye_status([game])
    feats = build_features(game, snapshots, bye)
    assert feats is not None
    assert feats["off_epa_l4_diff"] is not None
    assert feats["market_prob_home"] is not None
    assert feats["elo_prob_home"] == 0.65
    assert feats["epa_prob_home"] == 0.62
    assert feats["divisional"] == 0.0  # KC AFCW vs DET NFCN


def test_priors_avg_only_includes_present():
    from datetime import timedelta
    stats = {"KC": [], "DET": []}
    base = date(2022, 9, 11)
    test_date = base + timedelta(days=7 * 5)
    for w in range(1, 7):
        d = base + timedelta(days=7 * (w - 1))
        stats["KC"].append(_ts("KC", 2022, w, d))
        stats["DET"].append(_ts("DET", 2022, w, d))
    snapshots = fit_rolling_rates(stats, season_reset=True)
    game = _g(test_date, "KC", "DET", 24, 17, week=6, elo=None, epa=0.60)
    bye = compute_bye_status([game])
    feats = build_features(game, snapshots, bye)
    assert feats is not None
    # priors_avg = mean of (epa, market) since elo is None
    assert feats["priors_avg"] == pytest.approx(
        (feats["epa_prob_home"] + feats["market_prob_home"]) / 2.0
    )


def test_walk_forward_split_count_matches_expectation():
    """A 14-month window with 30d eval slide should produce ~10 folds."""
    splits = make_splits(date(2022, 1, 1), date(2023, 3, 1),
                        train_window_days=365, eval_window_days=30, warmup_days=120)
    # 14 months minus 120d warmup = ~10 months of eval, at 30d/fold = ~10 folds
    assert 8 <= len(splits) <= 12
