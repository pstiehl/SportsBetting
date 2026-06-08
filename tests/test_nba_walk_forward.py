"""Tests for the NBA walk-forward feature builder + simulator.

No network. All tests use synthetic ``GameRow`` data.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from flashcat.nba_features.feature_builder import (
    FEATURE_NAMES,
    GameRow,
    build_features,
    build_rolling_signals,
    feature_vector,
    normalize_team,
)
from flashcat.nba_features.model import (
    WalkForwardSplit,
    make_splits,
    walk_forward_evaluate,
)
from flashcat.nba_features.simulator import (
    NBA_LOSS_BUCKETS,
    _classify_loss_nba,
    _prob_to_decimal_with_hold,
    format_summary_table,
    settle_bets,
    summarize,
)


def _mk(
    d: date, home: str, away: str, *, home_won: int = 1,
    raptor: float | None = 0.55, elo: float | None = 0.56,
    bref: float | None = 0.54,
) -> GameRow:
    return GameRow(
        game_date=d,
        season=d.year if d.month <= 7 else d.year + 1,
        home=home, away=away,
        home_won=home_won,
        raptor_prob_home=raptor,
        elo_modern_prob_home=elo,
        bref_srs_prob_home=bref,
    )


# ---------------------------------------------------------------------------
# Team-code normalization
# ---------------------------------------------------------------------------


def test_normalize_team_aliases():
    assert normalize_team("BKN") == "BRK"
    assert normalize_team("BRK") == "BRK"
    assert normalize_team("PHX") == "PHO"
    assert normalize_team("PHO") == "PHO"
    assert normalize_team("CHA") == "CHO"
    assert normalize_team("LAL") == "LAL"
    assert normalize_team("  lal ") == "LAL"


# ---------------------------------------------------------------------------
# Walk-forward split assertions
# ---------------------------------------------------------------------------


def test_walk_forward_split_blocks_leakage():
    with pytest.raises(AssertionError):
        WalkForwardSplit(
            train_start=date(2022, 1, 1),
            train_end=date(2022, 6, 1),
            eval_start=date(2022, 6, 1),  # same day as train_end → leak
            eval_end=date(2022, 6, 30),
        )


def test_make_splits_strictly_separates_train_eval():
    splits = make_splits(
        date(2022, 1, 1), date(2022, 12, 31),
        train_window_days=180, eval_window_days=30, warmup_days=90,
    )
    assert len(splits) > 0
    for s in splits:
        assert s.train_end < s.eval_start
        assert s.train_start <= s.train_end
        assert s.eval_start <= s.eval_end


# ---------------------------------------------------------------------------
# Rolling-signal leakage gate
# ---------------------------------------------------------------------------


def test_rolling_signals_only_use_past():
    """Rolling features at game G must not depend on outcomes of G or later."""
    games = []
    d0 = date(2022, 1, 1)
    # Build a deterministic streak: LAL wins 5 then loses 5, alternating opponents.
    for i in range(20):
        home_won = 1 if i < 10 else 0
        # Alternate opponents to keep the rolling state per-team disjoint.
        opp = "BOS" if i % 2 == 0 else "NYK"
        games.append(_mk(d0 + timedelta(days=i), "LAL", opp, home_won=home_won))
    rolling = build_rolling_signals(games)

    # Game 0: LAL has no prior games → win_pct_l5 should be 0.5 (default).
    f0 = rolling[id(games[0])]
    assert f0["win_pct_l5_home"] == 0.5

    # Game 10 (first loss): LAL has just gone 10-0 → win_pct_l5_home = 1.0
    f10 = rolling[id(games[10])]
    assert f10["win_pct_l5_home"] == 1.0

    # Game 15: LAL has gone 0-5 in the last 5 → win_pct_l5_home = 0.0
    f15 = rolling[id(games[15])]
    assert f15["win_pct_l5_home"] == 0.0


def test_b2b_detection():
    games = []
    d0 = date(2022, 1, 1)
    games.append(_mk(d0, "LAL", "BOS", home_won=1))
    games.append(_mk(d0 + timedelta(days=1), "LAL", "NYK", home_won=0))  # B2B
    games.append(_mk(d0 + timedelta(days=4), "LAL", "MIA", home_won=1))  # 3 days rest
    rolling = build_rolling_signals(games)

    assert rolling[id(games[1])]["b2b_home"] == 1.0
    assert rolling[id(games[1])]["days_rest_home"] == 1.0
    assert rolling[id(games[2])]["b2b_home"] == 0.0
    assert rolling[id(games[2])]["days_rest_home"] == 3.0


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def test_build_features_returns_none_when_no_priors():
    g = GameRow(
        game_date=date(2022, 3, 1), season=2022,
        home="LAL", away="BOS", home_won=1,
        raptor_prob_home=None, elo_modern_prob_home=None,
        bref_srs_prob_home=None,
    )
    rolling = build_rolling_signals([g])
    assert build_features(g, rolling) is None


def test_feature_vector_order_matches_FEATURE_NAMES():
    g = _mk(date(2022, 3, 1), "LAL", "BOS")
    rolling = build_rolling_signals([g])
    f = build_features(g, rolling)
    assert f is not None
    v = feature_vector(f)
    assert len(v) == len(FEATURE_NAMES)
    # Spot-check first three entries are the per-source priors.
    assert v[0] == pytest.approx(g.raptor_prob_home)
    assert v[1] == pytest.approx(g.elo_modern_prob_home)
    assert v[2] == pytest.approx(g.bref_srs_prob_home)


def test_prior_consensus_and_dispersion():
    g = _mk(date(2022, 3, 1), "LAL", "BOS",
            raptor=0.60, elo=0.55, bref=0.50)
    rolling = build_rolling_signals([g])
    f = build_features(g, rolling)
    assert f is not None
    assert f["prior_consensus"] == pytest.approx(0.55)
    assert f["prior_dispersion"] == pytest.approx(0.10)
    assert f["raptor_vs_elo_disagree"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def test_prob_to_decimal_with_hold_basic():
    # 50% true prob with 4.5% hold → implied 52.25%, decimal ~1.913
    assert _prob_to_decimal_with_hold(0.50, 0.045) == pytest.approx(
        1.0 / (0.50 * 1.045), rel=1e-6
    )


def test_classify_loss_nba_pure_variance():
    bet = {"pick_home": True, "pick_prob": 0.51, "market_implied_prob": 0.50}
    assert _classify_loss_nba(bet, {}) == "pure_variance"


def test_classify_loss_nba_rest_disadvantage():
    bet = {"pick_home": True, "pick_prob": 0.60, "market_implied_prob": 0.58}
    feats = {
        "days_rest_diff": -2.0, "b2b_home": 1.0, "b2b_away": 0.0,
        "win_pct_diff_l10": 0.0, "prior_dispersion": 0.0,
    }
    assert _classify_loss_nba(bet, feats) == "rest_disadvantage"


def test_classify_loss_nba_pace_signal_wrong():
    bet = {"pick_home": True, "pick_prob": 0.65, "market_implied_prob": 0.62}
    feats = {
        "days_rest_diff": 0.0, "b2b_home": 0.0, "b2b_away": 0.0,
        "win_pct_diff_l10": 0.0, "prior_dispersion": 0.12,
    }
    assert _classify_loss_nba(bet, feats) == "pace_signal_wrong"


def test_classify_loss_nba_rolling_signal_wrong():
    bet = {"pick_home": True, "pick_prob": 0.65, "market_implied_prob": 0.62}
    feats = {
        "days_rest_diff": 0.0, "b2b_home": 0.0, "b2b_away": 0.0,
        "win_pct_diff_l10": 0.15, "prior_dispersion": 0.0,
    }
    assert _classify_loss_nba(bet, feats) == "rolling_signal_wrong"


def test_summarize_empty():
    out = summarize([])
    assert out["overall"]["n_bets"] == 0
    assert out["loss_buckets"] == {}


# ---------------------------------------------------------------------------
# End-to-end smoke: small synthetic walk-forward
# ---------------------------------------------------------------------------


def test_walk_forward_evaluate_smoke():
    # Build a year of synthetic NBA-shaped games for two teams alternating.
    # Real NBA: a team plays at most once per calendar day. Synthesize a
    # schedule where each (date, team) pair appears at most once.
    games = []
    teams = ["LAL", "BOS", "NYK", "DEN", "MIA", "GSW"]
    d = date(2022, 1, 1)
    # Pair teams into matchups; each day uses disjoint pairs.
    pairings_by_day = [
        [("LAL", "BOS"), ("NYK", "DEN"), ("MIA", "GSW")],
        [("BOS", "LAL"), ("DEN", "NYK"), ("GSW", "MIA")],
        [("LAL", "NYK"), ("BOS", "DEN"), ("MIA", "GSW")],
        [("NYK", "LAL"), ("DEN", "BOS"), ("GSW", "MIA")],
    ]
    for i in range(150):
        day_pairings = pairings_by_day[i % len(pairings_by_day)]
        for j, (h, a) in enumerate(day_pairings):
            raptor = 0.55 + 0.2 * (((i * 3 + j) % 7) - 3) / 6.0
            won = 1 if raptor > 0.55 else 0
            games.append(_mk(d + timedelta(days=i), h, a,
                             home_won=won, raptor=raptor, elo=raptor + 0.01,
                             bref=raptor - 0.01))
    rolling = build_rolling_signals(games)
    folds = walk_forward_evaluate(
        games, rolling,
        train_window_days=60, eval_window_days=15, warmup_days=30,
    )
    assert len(folds) > 0
    # Every fold respects the leakage gate (assertions inside).
    # Settling should produce some bets.
    bets, summary = settle_bets(folds, edge_gate=None)
    assert summary["n_bets"] > 0
    # Loss buckets, when present, must be from the documented vocabulary.
    for bucket in summary["loss_buckets"]:
        assert bucket in NBA_LOSS_BUCKETS
    # Format table should not raise.
    txt = format_summary_table(summary)
    assert "Walk-forward NBA backtest" in txt
