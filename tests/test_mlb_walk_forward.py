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
    PitcherForm,
    build_features,
    compute_pitcher_rest,
    feature_vector,
    fit_empirical_park_factor,
    fit_pitcher_form,
    fit_rolling_rates,
    load_538_mlb_games,
    strength_prior_home,
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
    h_pen: int = 3, a_pen: int = 4,
) -> GameRow:
    # Box-score pitching lines: home staff allowed ``as_`` runs, away staff
    # allowed ``hs`` runs (a coarse but self-consistent synthetic line).
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
        home_pitchers_used=h_pen,
        away_pitchers_used=a_pen,
        home_team_er=as_,
        away_team_er=hs,
        home_so_pitching=6,
        away_so_pitching=5,
        home_bb_pitching=2,
        away_bb_pitching=3,
        home_hr_allowed=1,
        away_hr_allowed=1,
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
    """build_features must return None when required features missing.

    Week-8: the required feature is the rolling strength prior
    (``prior_prob_home``), which needs L10 rolling rates on BOTH teams. 538's
    elo_prob_home is no longer required (the archive is dead).
    """
    games = _build_series(15)
    snaps = fit_rolling_rates(games)
    rest = compute_pitcher_rest(games)
    # Should produce features for game 11+ (l10 available).
    g = games[11]
    f = build_features(g, snaps, rest, {})
    assert f is not None
    for name in FEATURE_NAMES:
        assert name in f
    assert f["prior_prob_home"] is not None
    # An early game (index 3) has < 10 prior games -> no L10 -> no prior ->
    # rejected.
    g_early = games[3]
    f2 = build_features(g_early, snaps, rest, {})
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
# Week-8 expansion: pitcher form / bullpen / park / strength prior
# ---------------------------------------------------------------------------


def test_strength_prior_home_field_and_monotonic():
    """Prior gives home >50% at parity and rises with home run-diff edge."""
    # Equal rolling run diff -> only home-field advantage applies (>50%).
    p_even = strength_prior_home(0.0, 0.0)
    assert p_even is not None and 0.50 < p_even < 0.56
    # Home team much better -> higher prob; worse -> lower. Monotonic.
    p_up = strength_prior_home(2.0, -1.0)
    p_down = strength_prior_home(-2.0, 1.0)
    assert p_down < p_even < p_up
    # Missing sample -> None.
    assert strength_prior_home(None, 0.0) is None
    assert strength_prior_home(0.0, None) is None


def test_pitcher_form_strictly_prior():
    """Pitcher-form snapshot at a start reflects only earlier starts."""
    games = _build_series(12)  # same starter ids each game (homep001/awayp001)
    forms = fit_pitcher_form(games)
    # The first start has no prior data -> all windows None.
    g0 = games[0]
    f0 = forms[(g0.game_date, g0.home_pitcher_id)]
    assert f0.er_l(3) is None
    # A later start should have accumulated >=3 prior starts -> er_l(3) real.
    g_late = games[6]
    f_late = forms[(g_late.game_date, g_late.home_pitcher_id)]
    assert f_late.er_l(3) is not None
    # Home starter allowed away_score each game; _build_series alternates the
    # home team's runs-allowed between 3 (even i) and 6 (odd i).
    assert f_late.er_l(3) >= 0


def test_pitcher_form_season_reset():
    """Pitcher form resets across seasons."""
    g1 = _mk(date(2022, 9, 1), "BOS", "NYA", 5, 3, season=2022)
    g2 = _mk(date(2022, 9, 6), "BOS", "NYA", 5, 3, season=2022)
    g3 = _mk(date(2023, 4, 1), "BOS", "NYA", 5, 3, season=2023)
    forms = fit_pitcher_form([g1, g2, g3])
    # At the 2023 start, the prior-2022 lines must not carry over.
    f = forms[(g3.game_date, g3.home_pitcher_id)]
    assert len(f.er) == 0


def test_empirical_park_factor_prior_and_growth():
    """Empirical park factor is strictly-prior and moves toward observed runs."""
    # A high-scoring park: every game totals 15 runs.
    games = [
        _mk(date(2024, 4, 1) + timedelta(days=i), "BOS", "NYA", 8, 7, park="COORS")
        for i in range(40)
    ]
    pf = fit_empirical_park_factor(games, prior_runs=4.5, prior_weight=40)
    first = pf[(games[0].game_date, "COORS")]
    last = pf[(games[-1].game_date, "COORS")]
    # First game: no prior data -> equals the league prior total (9.0).
    assert first == pytest.approx(9.0, abs=1e-6)
    # By the last game the estimate has moved up toward 15 (observed).
    assert last > first
    assert 9.0 < last < 15.0


def test_week8_features_present_and_built():
    """build_features populates the Week-8 features when inputs are supplied."""
    games = _build_series(15)
    snaps = fit_rolling_rates(games)
    rest = compute_pitcher_rest(games)
    forms = fit_pitcher_form(games)
    parkf = fit_empirical_park_factor(games)
    g = games[12]
    f = build_features(g, snaps, rest, {}, pitcher_form=forms, park_factor_emp=parkf)
    assert f is not None
    for name in (
        "sp_er_l3_diff", "sp_kbb_l5_diff", "sp_hr_l5_diff",
        "bullpen_load_l3_diff", "park_run_env_emp", "prior_prob_home",
    ):
        assert name in f
    # Bullpen load diff should be a real number once >=3 prior games exist.
    assert f["bullpen_load_l3_diff"] is not None
    # Empirical park env should be populated (single park in fixture).
    assert f["park_run_env_emp"] is not None


def test_retrosheet_boxscore_parse_smoke():
    """If a Retrosheet cache is present, box-score fields must parse sanely."""
    from flashcat.mlb_features.feature_builder import RETROSHEET_CACHE
    from flashcat.mlb_features import load_retrosheet_games
    cache = RETROSHEET_CACHE / "gl2022.zip"
    if not cache.exists():
        pytest.skip("retrosheet gl2022.zip cache not present")
    games = load_retrosheet_games(2022)
    assert len(games) > 2000
    with_box = [g for g in games if g.home_pitchers_used is not None]
    assert len(with_box) > 2000
    g = with_box[0]
    # Sanity ranges on the extracted box-score fields.
    assert 1 <= g.home_pitchers_used <= 12
    assert 1 <= g.away_pitchers_used <= 12
    assert g.home_team_er is not None and g.home_team_er >= 0
    # Home staff's earned runs should be <= away team's runs scored
    # (unearned runs possible), and both are non-negative.
    assert g.away_score is None or g.home_team_er <= g.away_score + 5


# ---------------------------------------------------------------------------
# End-to-end smoke (Retrosheet cache preferred; 538 legacy skipped)
# ---------------------------------------------------------------------------


def test_end_to_end_smoke_retrosheet_2022():
    """Full pipeline on a 2022 Retrosheet window. Produces bets, no leakage."""
    from flashcat.mlb_features.feature_builder import RETROSHEET_CACHE
    from flashcat.mlb_features import load_retrosheet_games
    if not (RETROSHEET_CACHE / "gl2022.zip").exists():
        pytest.skip("retrosheet gl2022.zip cache not present")
    games = load_retrosheet_games(2022)
    games = [g for g in games if g.game_date <= date(2022, 7, 1)]
    assert len(games) > 1000
    snaps = fit_rolling_rates(games)
    rest = compute_pitcher_rest(games)
    forms = fit_pitcher_form(games)
    parkf = fit_empirical_park_factor(games)
    folds = walk_forward_evaluate(
        games, snaps, rest, {},
        pitcher_form=forms, park_factor_emp=parkf,
        train_window_days=60, eval_window_days=15, warmup_days=45,
    )
    assert len(folds) > 0
    assert any(f.n_picks > 0 for f in folds)
    bets, summary = simulate(folds)
    assert summary["overall"]["n_bets"] > 0
    clv = summary["overall"]["clv_proxy_pp"]
    assert -1.0 < clv < 1.0


# ---------------------------------------------------------------------------
# End-to-end smoke (uses 538 cache, no network) — legacy, skipped when gone
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
