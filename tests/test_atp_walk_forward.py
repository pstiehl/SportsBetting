"""Walk-forward harness sanity tests for the ATP feature-expansion harness.

Mirrors ``tests/test_nfl_walk_forward.py`` in structure: small synthetic
datasets, no network, no I/O. Verifies:

  * Canonicalization matches the tennis-data / backfill convention.
  * Rolling form snapshots reflect ONLY matches strictly before the match
    date (day-granular leakage gate), including the same-day double-header
    edge case.
  * Head-to-head is walk-forward (prior meetings only; 0.5 prior on first
    meeting; same-day rematch reads prior history only).
  * Walk-forward splits never overlap train and eval windows.
  * Required-feature gate (market + L10 rolling win%) is enforced.
  * Feature vector length matches FEATURE_NAMES and None-fill works.
  * Simulator returns symmetric pnl signs and uses real decimal odds.
  * Loss bucket classifier returns a valid bucket for every losing bet.
"""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.atp_features.feature_builder import (
    FEATURE_NAMES,
    MatchRow,
    RollingATPFeatures,
    _canonical_pair,
    _norm_name,
    build_features,
    compute_h2h,
    event_id,
    feature_vector,
    fit_rolling_rates,
)
from flashcat.atp_features.model import (
    WalkForwardSplit,
    make_splits,
    walk_forward_evaluate,
)
from flashcat.atp_features.simulator import (
    PRODUCTION_EDGE_GATE_PP,
    _classify_loss,
    simulate_flat_stake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _m(d, p1, p2, p1_won, *, surface="Hard", best_of=3, hrank=50, arank=60,
       hpts=1000.0, apts=800.0, hgames=12, agames=8, hsets=2, asets=0,
       market=None, bt=None, hdec=1.8, adec=2.0, tour="atp"):
    """Build a canonicalized MatchRow from (p1, p2) where p1_won says whether
    p1 won. Canonicalizes so home = alpha-first; flips scores/ranks/odds to
    the canonical orientation."""
    home, away, swap = _canonical_pair(p1, p2)
    if swap:
        home_won = not p1_won
        home_rank, away_rank = arank, hrank
        home_pts, away_pts = apts, hpts
        home_games, away_games = agames, hgames
        home_sets, away_sets = asets, hsets
        home_dec, away_dec = adec, hdec
    else:
        home_won = p1_won
        home_rank, away_rank = hrank, arank
        home_pts, away_pts = hpts, apts
        home_games, away_games = hgames, agames
        home_sets, away_sets = hsets, asets
        home_dec, away_dec = hdec, adec
    return MatchRow(
        event_id=event_id(tour, d, p1, p2),
        match_date=d,
        season=d.year,
        tour=tour,
        home=home,
        away=away,
        home_won=home_won,
        surface=surface,
        series="ATP250",
        round="1st Round",
        best_of=best_of,
        home_rank=home_rank,
        away_rank=away_rank,
        home_pts=home_pts,
        away_pts=away_pts,
        home_games=home_games,
        away_games=away_games,
        home_sets=home_sets,
        away_sets=away_sets,
        home_decimal=home_dec,
        away_decimal=away_dec,
        market_prob_home=market,
        rank_bt_prob_home=bt,
    )


# ---------------------------------------------------------------------------
# 1-3. Canonicalization
# ---------------------------------------------------------------------------

def test_canonical_pair_alpha_first():
    assert _canonical_pair("Alcaraz C.", "Zverev A.") == ("Alcaraz C.", "Zverev A.", False)
    home, away, swap = _canonical_pair("Zverev A.", "Alcaraz C.")
    assert home == "Alcaraz C." and away == "Zverev A." and swap is True


def test_norm_name_tennisdata_form():
    # "Sabalenka A." -> "sabalenka a"
    assert _norm_name("Sabalenka A.") == "sabalenka a"


def test_event_id_stable_regardless_of_order():
    a = event_id("atp", date(2023, 6, 1), "Nadal R.", "Federer R.")
    b = event_id("atp", date(2023, 6, 1), "Federer R.", "Nadal R.")
    assert a == b  # canonicalized identically


# ---------------------------------------------------------------------------
# 4-6. Rolling form — leakage gate
# ---------------------------------------------------------------------------

def test_rolling_snapshot_excludes_current_and_future_matches():
    """Snapshot at match N reflects only matches strictly before its date."""
    matches = [
        _m(date(2023, 1, 2), "Aaa X.", "Bbb Y.", True),
        _m(date(2023, 1, 9), "Aaa X.", "Ccc Z.", True),
        _m(date(2023, 1, 16), "Aaa X.", "Ddd W.", False),
    ]
    snaps = fit_rolling_rates(matches)
    # The 3rd match snapshot for "Aaa X." should reflect 2 prior wins.
    m3 = matches[2]
    home = m3.home if m3.home == "Aaa X." else m3.away
    snap = snaps[(m3.event_id, "Aaa X.")]
    assert list(snap.wins) == [1, 1]
    assert snap.last_match_date == date(2023, 1, 9)


def test_rolling_snapshot_same_day_double_header_no_leak():
    """Two matches for the same player on the same date must both snapshot
    against pre-date state (neither sees the other)."""
    matches = [
        _m(date(2023, 1, 2), "Aaa X.", "Bbb Y.", True),   # prior win
        _m(date(2023, 1, 9), "Aaa X.", "Ccc Z.", True),   # same-day match 1
        _m(date(2023, 1, 9), "Aaa X.", "Ddd W.", False),  # same-day match 2
    ]
    snaps = fit_rolling_rates(matches)
    s1 = snaps[(matches[1].event_id, "Aaa X.")]
    s2 = snaps[(matches[2].event_id, "Aaa X.")]
    # Both same-day snapshots reflect exactly one prior win (Jan 2), not each
    # other. And last_match_date is strictly before Jan 9.
    assert list(s1.wins) == [1]
    assert list(s2.wins) == [1]
    assert s1.last_match_date == date(2023, 1, 2)
    assert s2.last_match_date == date(2023, 1, 2)


def test_rolling_win_pct_insufficient_sample_returns_none():
    f = RollingATPFeatures()
    f.wins.extend([1, 0, 1])
    assert f.win_pct(10) is None
    assert f.win_pct(3) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 7-8. Head-to-head walk-forward
# ---------------------------------------------------------------------------

def test_h2h_first_meeting_is_prior_half():
    matches = [_m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True)]
    h2h = compute_h2h(matches)
    assert h2h[matches[0].event_id] == 0.5


def test_h2h_accumulates_prior_meetings_only():
    matches = [
        _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True),   # home wins
        _m(date(2023, 6, 1), "Aaa X.", "Bbb Y.", True),   # home wins again
        _m(date(2023, 7, 1), "Aaa X.", "Bbb Y.", False),  # home loses
    ]
    h2h = compute_h2h(matches)
    # 1st: no prior -> 0.5. 2nd: 1 prior home win -> 1.0. 3rd: 2 prior home wins -> 1.0
    assert h2h[matches[0].event_id] == 0.5
    assert h2h[matches[1].event_id] == pytest.approx(1.0)
    assert h2h[matches[2].event_id] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9-10. Walk-forward splits
# ---------------------------------------------------------------------------

def test_walk_forward_split_no_leakage():
    splits = make_splits(date(2022, 1, 1), date(2024, 12, 31),
                         train_window_days=365, eval_window_days=30, warmup_days=120)
    assert len(splits) > 0
    for s in splits:
        assert s.train_end < s.eval_start
        assert s.eval_end >= s.eval_start
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
# 11-13. Feature gate + vector
# ---------------------------------------------------------------------------

def test_build_features_returns_none_when_market_missing():
    m = _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True, market=None)
    f = build_features(m, rolling={}, h2h={})
    assert f is None


def test_build_features_returns_none_when_rolling_l10_missing():
    m = _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True, market=0.6)
    # Empty rolling state -> no L10 win% -> None even though market present.
    f = build_features(m, rolling={}, h2h={})
    assert f is None


def test_feature_vector_length_and_fill():
    feats = {name: None for name in FEATURE_NAMES}
    v = feature_vector(feats, fill_value=0.0)
    assert len(v) == len(FEATURE_NAMES)
    assert all(x == 0.0 for x in v)
    v2 = feature_vector(feats, fill_value=-1.0)
    assert all(x == -1.0 for x in v2)


# ---------------------------------------------------------------------------
# 14-16. Loss bucket classifier
# ---------------------------------------------------------------------------

def test_classify_loss_pure_variance():
    bet = {"pick_prob": 0.51, "market_implied_prob": 0.50, "edge_pp": 0.01, "pick_home": True}
    assert _classify_loss(bet, {}, {}) == "pure_variance"


def test_classify_loss_favorite_upset():
    bet = {"pick_prob": 0.80, "market_implied_prob": 0.75, "edge_pp": 0.05, "pick_home": True}
    assert _classify_loss(bet, {}, {}) == "favorite_upset"


def test_classify_loss_generic_fallback():
    bet = {"pick_prob": 0.60, "market_implied_prob": 0.55, "edge_pp": 0.05, "pick_home": True}
    feats = {"surface_win_pct_l20_diff": 0.0, "rank_points_log_ratio": 0.0,
             "rest_days_diff": 0.0, "h2h_home_share": 0.5, "best_of_5": 0.0}
    assert _classify_loss(bet, feats, {}) == "generic"


# ---------------------------------------------------------------------------
# 17-19. End-to-end (synthetic, no I/O)
# ---------------------------------------------------------------------------

def _long_history(player_a="Aaa X.", player_b="Bbb Y.", n=14, start=date(2023, 1, 2)):
    """Build n weekly matches so L10 windows populate. Returns matches list."""
    from datetime import timedelta
    out = []
    d = start
    for i in range(n):
        # Alternate opponents so both A and B accrue history; A wins ~most.
        out.append(_m(d, player_a, f"Opp{i%5} Z.", i % 3 != 0, market=0.6))
        out.append(_m(d, player_b, f"Opq{i%5} Y.", i % 2 == 0, market=0.55))
        d = d + timedelta(days=7)
    return out


def test_e2e_build_features_with_rolling_history():
    from datetime import timedelta
    hist = _long_history()
    test_date = date(2023, 1, 2) + timedelta(days=7 * 20)
    target = _m(test_date, "Aaa X.", "Bbb Y.", True, market=0.62, bt=0.58)
    matches = hist + [target]
    snaps = fit_rolling_rates(matches)
    h2h = compute_h2h(matches)
    feats = build_features(target, snaps, h2h)
    assert feats is not None
    assert feats["market_prob_home"] == 0.62
    assert feats["win_pct_l10_diff"] is not None
    assert -1.0 <= feats["win_pct_l10_diff"] <= 1.0
    assert feats["rank_bt_minus_market_pp"] == pytest.approx(0.58 - 0.62)


def test_e2e_walk_forward_and_simulate_sign_invariants():
    """Full pipeline on synthetic data: every won bet has pnl > 0, every
    lost bet has pnl == -stake, ROI is bounded, buckets are valid."""
    from datetime import timedelta
    # Build ~18 months of dense matches so at least one fold trains+evals.
    matches = []
    d0 = date(2022, 1, 3)
    for i in range(220):
        d = d0 + timedelta(days=2 * i)
        winner = i % 2 == 0
        matches.append(_m(d, "Aaa X.", "Bbb Y.", winner, market=0.55 if winner else 0.45,
                          bt=0.55, hdec=1.9, adec=2.0))
    snaps = fit_rolling_rates(matches)
    h2h = compute_h2h(matches)
    folds = walk_forward_evaluate(matches, snaps, h2h,
                                  train_window_days=365, eval_window_days=30, warmup_days=120)
    bets, summary = simulate_flat_stake(folds, edge_gate_pp=None)
    # Sign invariants.
    for b in bets:
        if b.won:
            assert b.pnl > 0
        else:
            assert b.pnl == pytest.approx(-100.0)
        assert b.loss_bucket is None if b.won else b.loss_bucket is not None
    if bets:
        assert -1.0 <= summary["roi"] <= 5.0
        assert 0.0 <= summary["win_rate"] <= 1.0
        # Loss buckets only count losers.
        assert sum(summary["loss_buckets"].values()) == sum(1 for b in bets if not b.won)


def test_production_edge_gate_constant_is_3pp():
    assert PRODUCTION_EDGE_GATE_PP == 0.03
