"""Tests for the MLB walk-forward feature builder + simulator.

We do NOT hit the network. All tests use synthetic ``GameRow`` data or
the committed ``data/cache/538_mlb_elo.csv`` cache.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from flashcat.mlb_features.feature_builder import (
    FEATURE_NAMES,
    GameRow,
    build_features,
    compute_pitcher_rest,
    feature_vector,
    fit_rolling_rates,
    load_538_mlb_games,
)
from flashcat.mlb_features.model import (
    WalkForwardSplit,
    make_splits,
    walk_forward_evaluate,
)
from flashcat.mlb_features.simulator import (
    _prob_to_decimal_with_hold,
    format_summary_table,
    simulate,
    summarize,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _mk(
    d: date, home: str, away: str, hs: int, as_: int,
    *, elo_p_home: float = 0.55, rating_p_home: float = 0.56,
    rgs_h: float = 1.0, rgs_a: float = 0.5,
    home_pid: str = "homep001", away_pid: str = "awayp001",
    park: str = "BOS07", season: int | None = None,
) -> GameRow:
    return GameRow(
        game_date=d,
        season=season or d.year,
        home=home, away=away,
        home_score=hs, away_score=as_,
        elo_prob_home=elo_p_home,
        rating_prob_home=rating_p_home,
        pitcher_rgs_home=rgs_h,
        pitcher_rgs_away=rgs_a,
        pitcher_adj_home=rgs_h * 0.5,
        pitcher_adj_away=rgs_a * 0.5,
        park_id=park,
        day_night="N",
        home_pitcher_id=home_pid,
        away_pitcher_id=away_pid,
    )


def _build_series(n_per_team: int = 25) -> list[GameRow]:
    """A 2-team series so rolling features can populate L20 windows."""
    games: list[GameRow] = []
    start = date(2024, 4, 1)
    for i in range(n_per_team):
        d = start + timedelta(days=i)
        # Home wins on even i, away wins on odd
        if i % 2 == 0:
            games.append(_mk(d, "BOS", "NYA", 5, 3))
        else:
            games.append(_mk(d, "BOS", "NYA", 2, 6))
    return games


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def test_load_538_cache_smoke():
    """The 538 cache (if present locally) should yield 2022+2023 games.

    Skipped on fresh-clone CI where data/cache/538_mlb_elo.csv is not yet
    populated. The production ``flashcat all`` path downloads this CSV
    once and caches it; once that's run the test exercises it.
    """
    from flashcat.mlb_features.feature_builder import ELO_538_CACHE
    if not ELO_538_CACHE.exists():
        pytest.skip("data/cache/538_mlb_elo.csv not present; run flashcat all first")
    games = load_538_mlb_games(start=date(2022, 1, 1), end=date(2023, 12, 31))
    # Two full MLB regular seasons ≈ 4860 games.
    assert len(games) >= 4500, f"unexpectedly small 538 cache: {len(games)}"
    # All in window
    for g in games[:20]:
        assert date(2022, 1, 1) <= g.game_date <= date(2023, 12, 31)
        assert g.home and g.away
        assert g.home != g.away


def test_rolling_rates_walk_forward_strict():
    """Snapshot at date D must reflect only games strictly before D."""
    games = _build_series(20)
    snaps = fit_rolling_rates(games)
    # For each game, snapshot at that date should not include that game.
    by_date = {}
    for g in games:
        by_date[g.game_date] = g
    for d, snap in snaps.items():
        for team, feats in snap.items():
            if feats.last_game_date is None:
                continue
            assert feats.last_game_date < d, (
                f"leakage: {team} last_game_date {feats.last_game_date} on snapshot {d}"
            )


def test_rolling_rates_l10():
    games = _build_series(15)
    snaps = fit_rolling_rates(games)
    # After 10 games each, L10 for BOS should reflect alternating 5/3 and 2/6.
    # BOS RS L10 = mean of last 10 BOS runs scored. Pattern: 5,2,5,2,5,2,5,2,5,2 = 35/10 = 3.5
    final_date = games[-1].game_date
    snap = snaps[final_date]
    bos = snap["BOS"]
    rs_l10 = bos.rs(10)
    # Last 10 BOS games' RS: we built 14 prior to the 15th game; the last 10
    # are games 5..14. Even-indexed -> 5 RS; odd -> 2 RS. Indexes 5..14 contain
    # 5 evens and 5 odds → mean = (5*5 + 5*2)/10 = 3.5.
    assert rs_l10 == pytest.approx(3.5, abs=0.01), rs_l10


def test_season_reset():
    """Rolling features reset at season boundaries."""
    g1 = _mk(date(2022, 9, 30), "BOS", "NYA", 9, 1, season=2022)
    g2 = _mk(date(2022, 10, 1), "BOS", "NYA", 9, 1, season=2022)
    g3 = _mk(date(2022, 10, 2), "BOS", "NYA", 9, 1, season=2022)
    g4 = _mk(date(2023, 4, 1), "BOS", "NYA", 0, 10, season=2023)
    snaps = fit_rolling_rates([g1, g2, g3, g4])
    snap = snaps[date(2023, 4, 1)]
    # BOS at 2023-04-01 should NOT carry the 9 RS/game from 2022.
    bos = snap["BOS"]
    # Since 2023 season hasn't started yet (it's THE first game), there are 0 games.
    assert len(bos.runs_scored) == 0


def test_pitcher_rest_first_appearance():
    games = _build_series(5)
    rest = compute_pitcher_rest(games)
    # First date should give rest=6 (league avg fallback).
    first_d = games[0].game_date
    pid = games[0].home_pitcher_id
    assert rest[(first_d, pid)] == 6


def test_build_features_required_gate():
    """build_features must return None when required features missing."""
    games = _build_series(15)
    snaps = fit_rolling_rates(games)
    rest = compute_pitcher_rest(games)
    # Should produce features for game 11+ (l10 available)
    g = games[11]
    f = build_features(g, snaps, rest, {})
    assert f is not None
    for name in FEATURE_NAMES:
        assert name in f
    # Take a game with no elo_prob_home — should be rejected.
    g_bad = _mk(g.game_date, "BOS", "NYA", 5, 3, elo_p_home=None)
    f2 = build_features(g_bad, snaps, rest, {})
    assert f2 is None


def test_feature_vector_order():
    """feature_vector returns values in FEATURE_NAMES order, filling None as 0."""
    feats = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    v = feature_vector(feats)
    assert len(v) == len(FEATURE_NAMES)
    for i, val in enumerate(v):
        assert val == pytest.approx(float(i))
    # Inject a None — should get fill_value.
    feats["pitcher_rest_diff"] = None
    v2 = feature_vector(feats, fill_value=99.0)
    idx = FEATURE_NAMES.index("pitcher_rest_diff")
    assert v2[idx] == 99.0


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------


def test_make_splits_leakage_gate():
    """Every split must have train_end < eval_start (asserted in __post_init__)."""
    splits = make_splits(
        date(2022, 4, 1), date(2022, 10, 1),
        train_window_days=365, eval_window_days=30, warmup_days=60,
    )
    assert len(splits) > 0
    for s in splits:
        assert s.train_end < s.eval_start
        assert s.eval_start <= s.eval_end


def test_make_splits_invalid_raises():
    """If we attempt to construct an invalid WalkForwardSplit, post-init asserts."""
    with pytest.raises(AssertionError):
        WalkForwardSplit(
            train_start=date(2022, 1, 1),
            train_end=date(2022, 6, 30),
            eval_start=date(2022, 6, 30),  # same day → must be strictly after
            eval_end=date(2022, 7, 30),
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_prob_to_decimal_with_hold_monotonic():
    # Higher true prob → lower decimal odds.
    d1 = _prob_to_decimal_with_hold(0.4)
    d2 = _prob_to_decimal_with_hold(0.6)
    assert d1 > d2
    # Reasonable range.
    assert 1.5 < d1 < 3.0
    assert 1.0 < d2 < 2.0


def test_simulate_no_gate_bets_every_game():
    """With edge_gate=None we bet every fold pred."""
    from flashcat.mlb_features.model import FoldResult, WalkForwardSplit
    split = WalkForwardSplit(
        train_start=date(2022, 1, 1), train_end=date(2022, 5, 31),
        eval_start=date(2022, 6, 1), eval_end=date(2022, 6, 30),
    )
    preds = [
        {"game_date": "2022-06-15", "home": "BOS", "away": "NYA",
         "home_prob": 0.65, "rating_prob_home": 0.60,
         "elo_prob_home": 0.58, "home_won": 1, "features": {}},
        {"game_date": "2022-06-16", "home": "BOS", "away": "NYA",
         "home_prob": 0.40, "rating_prob_home": 0.45,
         "elo_prob_home": 0.50, "home_won": 0, "features": {}},
    ]
    fold = FoldResult(
        split=split, n_train=200, n_eval=2, n_picks=2,
        coef=[0.0] * len(FEATURE_NAMES), intercept=0.0,
        log_loss=0.6, accuracy=1.0, brier=0.25,
        predictions=preds,
    )
    bets, summary = simulate([fold])
    assert len(bets) == 2
    # Game 1: pick home (0.65 > 0.5), home wins → won
    assert bets[0].won is True
    # Game 2: pick away (home_prob 0.40 < 0.5), away wins (home_won=0) → won
    assert bets[1].won is True
    # ROI > 0 when both bets win.
    assert summary["overall"]["roi"] > 0


def test_simulate_loss_buckets_classified():
    """Losing bets get bucketed; winning bets don't get a bucket."""
    from flashcat.mlb_features.model import FoldResult, WalkForwardSplit
    split = WalkForwardSplit(
        train_start=date(2022, 1, 1), train_end=date(2022, 5, 31),
        eval_start=date(2022, 6, 1), eval_end=date(2022, 6, 30),
    )
    # Strong-pitcher signal that lost.
    preds = [
        {"game_date": "2022-06-15", "home": "BOS", "away": "NYA",
         "home_prob": 0.65, "rating_prob_home": 0.55,
         "elo_prob_home": 0.55, "home_won": 0,  # we picked home, home lost
         "features": {"pitcher_rgs_diff": 1.5, "run_diff_l10_diff": 0.0}},
        # Coinflip pure-variance loss
        {"game_date": "2022-06-16", "home": "BOS", "away": "NYA",
         "home_prob": 0.51, "rating_prob_home": 0.51,
         "elo_prob_home": 0.51, "home_won": 0,  # we picked home, home lost
         "features": {}},
    ]
    fold = FoldResult(
        split=split, n_train=200, n_eval=2, n_picks=2,
        coef=[0.0] * len(FEATURE_NAMES), intercept=0.0,
        log_loss=0.6, accuracy=0.5, brier=0.25,
        predictions=preds,
    )
    bets, summary = simulate([fold])
    assert all(not b.won for b in bets)
    buckets = summary["loss_buckets"]
    assert buckets.get("pitcher_signal_wrong", 0) == 1
    assert buckets.get("pure_variance", 0) == 1


def test_summary_table_formats_without_crashing():
    bets, summary = simulate([])
    out = format_summary_table(summary)
    assert "Walk-forward" in out


# ---------------------------------------------------------------------------
# End-to-end smoke (uses 538 cache, no network)
# ---------------------------------------------------------------------------


def test_end_to_end_smoke_2022():
    """Full pipeline on a short 2022 window. Must run < ~30s, produce bets."""
    from flashcat.mlb_features.feature_builder import ELO_538_CACHE
    if not ELO_538_CACHE.exists():
        pytest.skip("data/cache/538_mlb_elo.csv not present; run flashcat all first")
    games = load_538_mlb_games(start=date(2022, 1, 1), end=date(2022, 7, 1))
    assert len(games) > 1000, "expected ~1300 games for 2022-04..07"
    snaps = fit_rolling_rates(games)
    rest = compute_pitcher_rest(games)
    folds = walk_forward_evaluate(
        games, snaps, rest, {},
        train_window_days=60, eval_window_days=15, warmup_days=45,
    )
    assert len(folds) > 0
    # All folds should have at least some predictions
    assert any(f.n_picks > 0 for f in folds)
    bets, summary = simulate(folds)
    assert summary["overall"]["n_bets"] > 0
    # CLV proxy is a number in (-1, 1)
    clv = summary["overall"]["clv_proxy_pp"]
    assert -1.0 < clv < 1.0
