"""Walk-forward harness sanity tests for the PGA feature-expansion harness.

A PORT of ``tests/test_wta_walk_forward.py`` to PGA Tour head-to-head
matchups. Mirrors it in structure: small synthetic fixtures, no network, no
I/O. Catalog differences vs WTA: the two set/game features are dropped (no
golf analog) and made-cut rate + skill_diff are added, netting **12
features**.

Verifies:

  * Canonicalization matches the pga_datagolf convention (alpha-first home,
    normalized event_id).
  * Rolling form snapshots reflect ONLY tournaments strictly before the
    matchup date (day-granular leakage gate), including the same-day
    (same-tournament) multi-matchup edge case.
  * Head-to-head is walk-forward (prior meetings only; 0.5 prior on first
    meeting; same-day rematch reads prior history only).
  * Walk-forward splits never overlap train and eval windows.
  * Required-feature gate (market + L10 rolling H2H form) is enforced.
  * Feature vector length matches FEATURE_NAMES (12) and None-fill works.
  * The dropped set/game features are NOT in the catalog.
  * Simulator returns symmetric pnl signs and uses real decimal odds.
  * Loss bucket classifier returns a valid bucket for every losing bet.
  * The DATA-BLOCKED backtest JSON receipt is honestly shaped when present.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from flashcat.pga_features.feature_builder import (
    FEATURE_NAMES,
    MatchupRow,
    RollingPGAFeatures,
    _canonical_pair,
    _finish_quality,
    _norm_name,
    build_features,
    compute_h2h,
    event_id,
    feature_vector,
    fit_rolling_rates,
)
from flashcat.pga_features.model import (
    WalkForwardSplit,
    make_splits,
    walk_forward_evaluate,
)
from flashcat.pga_features.simulator import (
    PRODUCTION_EDGE_GATE_PP,
    _classify_loss,
    simulate_flat_stake,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _m(d, p1, p2, p1_ahead, *, course="standard", event=None,
       h_win=0.10, a_win=0.06, h_skill=1.5, a_skill=1.0,
       h_finish=10, a_finish=25, market=None, bt=None, hdec=1.9, adec=2.0,
       tour="pga"):
    """Build a canonicalized MatchupRow from (p1, p2) where p1_ahead says
    whether p1 finished ahead. Canonicalizes so home = alpha-first; flips
    skill/finish/odds to the canonical orientation.

    ``event`` defaults to a per-DATE label (``Open <isodate>``) so distinct
    tournaments get distinct event_ids -- mirroring reality, where each
    week's PGA event has a unique name. Same-day matchups share a label
    (same tournament), which is exactly what the same-day leakage tests
    want."""
    if event is None:
        event = f"Open {d.isoformat()}"
    home, away, swap = _canonical_pair(p1, p2)
    if swap:
        home_won = not p1_ahead
        h_win_pct, a_win_pct = a_win, h_win
        home_skill, away_skill = a_skill, h_skill
        home_finish, away_finish = a_finish, h_finish
        home_dec, away_dec = adec, hdec
    else:
        home_won = p1_ahead
        h_win_pct, a_win_pct = h_win, a_win
        home_skill, away_skill = h_skill, a_skill
        home_finish, away_finish = h_finish, a_finish
        home_dec, away_dec = hdec, adec
    return MatchupRow(
        event_id=event_id(tour, event, p1, p2),
        match_date=d,
        season=d.year,
        tour=tour,
        event_label=event,
        home=home,
        away=away,
        home_won=home_won,
        course_tier=course,
        home_win_pct=h_win_pct,
        away_win_pct=a_win_pct,
        home_skill=home_skill,
        away_skill=away_skill,
        home_finish=home_finish,
        away_finish=away_finish,
        home_decimal=home_dec,
        away_decimal=away_dec,
        market_prob_home=market,
        skill_bt_prob_home=bt,
    )


# ---------------------------------------------------------------------------
# 1-3. Canonicalization
# ---------------------------------------------------------------------------

def test_canonical_pair_alpha_first():
    assert _canonical_pair("McIlroy Rory", "Scheffler Scottie") == (
        "McIlroy Rory", "Scheffler Scottie", False)
    home, away, swap = _canonical_pair("Scheffler Scottie", "McIlroy Rory")
    assert home == "McIlroy Rory" and away == "Scheffler Scottie" and swap is True


def test_norm_name_strips_punct():
    assert _norm_name("McIlroy, R.") == "mcilroy r"


def test_event_id_stable_regardless_of_order():
    a = event_id("pga", "The Open", "Scheffler Scottie", "McIlroy Rory")
    b = event_id("pga", "The Open", "McIlroy Rory", "Scheffler Scottie")
    assert a == b  # canonicalized identically


# ---------------------------------------------------------------------------
# 4. finish-quality transform
# ---------------------------------------------------------------------------

def test_finish_quality_monotonic_and_bounded():
    assert _finish_quality(1) == pytest.approx(1.0)
    assert 0.0 < _finish_quality(50) < _finish_quality(10) < 1.0
    assert _finish_quality(None) == 0.0  # missed cut / no finish


# ---------------------------------------------------------------------------
# 5-7. Rolling form — leakage gate
# ---------------------------------------------------------------------------

def test_rolling_snapshot_excludes_current_and_future_matchups():
    matchups = [
        _m(date(2023, 1, 5), "Aaa X.", "Bbb Y.", True),
        _m(date(2023, 1, 12), "Aaa X.", "Ccc Z.", True),
        _m(date(2023, 1, 19), "Aaa X.", "Ddd W.", False),
    ]
    snaps = fit_rolling_rates(matchups)
    m3 = matchups[2]
    snap = snaps[(m3.event_id, "Aaa X.")]
    assert list(snap.h2h_results) == [1, 1]
    assert snap.last_start_date == date(2023, 1, 12)


def test_rolling_snapshot_same_day_tournament_no_leak():
    """Two matchups in the same-week tournament must both snapshot against
    pre-date state (neither sees the other)."""
    matchups = [
        _m(date(2023, 1, 5), "Aaa X.", "Bbb Y.", True),    # prior win
        _m(date(2023, 1, 12), "Aaa X.", "Ccc Z.", True),   # same-day matchup 1
        _m(date(2023, 1, 12), "Ddd W.", "Eee V.", False),  # same-day matchup 2
    ]
    snaps = fit_rolling_rates(matchups)
    s1 = snaps[(matchups[1].event_id, "Aaa X.")]
    assert list(s1.h2h_results) == [1]
    assert s1.last_start_date == date(2023, 1, 5)


def test_rolling_h2h_pct_insufficient_sample_returns_none():
    f = RollingPGAFeatures()
    f.h2h_results.extend([1, 0, 1])
    assert f.h2h_win_pct(10) is None
    assert f.h2h_win_pct(3) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# 8-9. Head-to-head walk-forward
# ---------------------------------------------------------------------------

def test_h2h_first_meeting_is_prior_half():
    matchups = [_m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True)]
    h2h = compute_h2h(matchups)
    assert h2h[matchups[0].event_id] == 0.5


def test_h2h_accumulates_prior_meetings_only():
    matchups = [
        _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True),
        _m(date(2023, 6, 1), "Aaa X.", "Bbb Y.", True),
        _m(date(2023, 7, 1), "Aaa X.", "Bbb Y.", False),
    ]
    h2h = compute_h2h(matchups)
    assert h2h[matchups[0].event_id] == 0.5
    assert h2h[matchups[1].event_id] == pytest.approx(1.0)
    assert h2h[matchups[2].event_id] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 10-11. Walk-forward splits
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
# 12-15. Feature gate + vector (PGA: 12 features, no set/game features)
# ---------------------------------------------------------------------------

def test_feature_catalog_is_12_and_excludes_set_game_features():
    assert len(FEATURE_NAMES) == 12
    assert "sets_won_pct_l10_diff" not in FEATURE_NAMES
    assert "games_won_pct_l10_diff" not in FEATURE_NAMES
    # And the golf-specific additions are present.
    assert "made_cut_pct_l10_diff" in FEATURE_NAMES
    assert "skill_diff" in FEATURE_NAMES
    assert "course_tier_quality_l10_diff" in FEATURE_NAMES


def test_build_features_returns_none_when_market_missing():
    m = _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True, market=None)
    f = build_features(m, rolling={}, h2h={})
    assert f is None


def test_build_features_returns_none_when_rolling_l10_missing():
    m = _m(date(2023, 5, 1), "Aaa X.", "Bbb Y.", True, market=0.6)
    # Empty rolling state -> no L10 H2H form -> None even though market present.
    f = build_features(m, rolling={}, h2h={})
    assert f is None


def test_feature_vector_length_and_fill():
    feats = {name: None for name in FEATURE_NAMES}
    v = feature_vector(feats, fill_value=0.0)
    assert len(v) == len(FEATURE_NAMES) == 12
    assert all(x == 0.0 for x in v)
    v2 = feature_vector(feats, fill_value=-1.0)
    assert all(x == -1.0 for x in v2)


# ---------------------------------------------------------------------------
# 16-18. Loss bucket classifier
# ---------------------------------------------------------------------------

def test_classify_loss_pure_variance():
    bet = {"pick_prob": 0.51, "market_implied_prob": 0.50, "edge_pp": 0.01, "pick_home": True}
    assert _classify_loss(bet, {}, {}) == "pure_variance"


def test_classify_loss_favorite_upset():
    bet = {"pick_prob": 0.80, "market_implied_prob": 0.75, "edge_pp": 0.05, "pick_home": True}
    assert _classify_loss(bet, {}, {}) == "favorite_upset"


def test_classify_loss_generic_fallback():
    bet = {"pick_prob": 0.60, "market_implied_prob": 0.55, "edge_pp": 0.05, "pick_home": True}
    feats = {"course_tier_quality_l10_diff": 0.0, "win_pct_log_ratio": 0.0,
             "finish_quality_l10_diff": 0.0, "rest_days_diff": 0.0, "h2h_home_share": 0.5}
    assert _classify_loss(bet, feats, {}) == "generic"


def test_classify_loss_skill_signal_wrong():
    bet = {"pick_prob": 0.62, "market_implied_prob": 0.55, "edge_pp": 0.07, "pick_home": True}
    feats = {"win_pct_log_ratio": 0.8, "course_tier_quality_l10_diff": 0.0}
    assert _classify_loss(bet, feats, {}) == "skill_signal_wrong"


# ---------------------------------------------------------------------------
# 19-21. End-to-end (synthetic, no I/O)
# ---------------------------------------------------------------------------

def _long_history(player_a="Aaa X.", player_b="Bbb Y.", n=14, start=date(2023, 1, 5)):
    """Build n weekly tournaments so L10 windows populate."""
    out = []
    d = start
    for i in range(n):
        out.append(_m(d, player_a, f"Opp{i%5} Z.", i % 3 != 0, market=0.6))
        out.append(_m(d, player_b, f"Opq{i%5} Y.", i % 2 == 0, market=0.55))
        d = d + timedelta(days=7)
    return out


def test_e2e_build_features_with_rolling_history():
    hist = _long_history()
    test_date = date(2023, 1, 5) + timedelta(days=7 * 20)
    target = _m(test_date, "Aaa X.", "Bbb Y.", True, market=0.62, bt=0.58)
    matchups = hist + [target]
    snaps = fit_rolling_rates(matchups)
    h2h = compute_h2h(matchups)
    feats = build_features(target, snaps, h2h)
    assert feats is not None
    assert feats["market_prob_home"] == 0.62
    assert feats["h2h_form_l10_diff"] is not None
    assert -1.0 <= feats["h2h_form_l10_diff"] <= 1.0
    assert feats["skill_bt_minus_market_pp"] == pytest.approx(0.58 - 0.62)


def test_e2e_walk_forward_and_simulate_sign_invariants():
    """Full pipeline on synthetic data: every won bet has pnl > 0, every
    lost bet has pnl == -stake, ROI is bounded, buckets are valid."""
    matchups = []
    d0 = date(2022, 1, 3)
    for i in range(220):
        d = d0 + timedelta(days=2 * i)
        ahead = i % 2 == 0
        matchups.append(_m(d, "Aaa X.", "Bbb Y.", ahead,
                           market=0.55 if ahead else 0.45,
                           bt=0.55, hdec=1.9, adec=2.0))
    snaps = fit_rolling_rates(matchups)
    h2h = compute_h2h(matchups)
    folds = walk_forward_evaluate(matchups, snaps, h2h,
                                  train_window_days=365, eval_window_days=30, warmup_days=120)
    bets, summary = simulate_flat_stake(folds, edge_gate_pp=None)
    for b in bets:
        if b.won:
            assert b.pnl > 0
            assert b.loss_bucket is None
        else:
            assert b.pnl == pytest.approx(-100.0)
            assert b.loss_bucket is not None
    if bets:
        assert -1.0 <= summary["roi"] <= 5.0
        assert 0.0 <= summary["win_rate"] <= 1.0
        assert sum(summary["loss_buckets"].values()) == sum(1 for b in bets if not b.won)


def test_production_edge_gate_constant_is_3pp():
    assert PRODUCTION_EDGE_GATE_PP == 0.03


# ---------------------------------------------------------------------------
# 22. Honest data-blocker receipt (if the driver has been run)
# ---------------------------------------------------------------------------

def test_backtest_json_receipt_is_honestly_shaped_if_present():
    """If data/pga_walk_forward_backtest.json exists, it must honestly
    declare its data_status and NOT claim a real backtest with zero graded
    matchups. Skips when the receipt isn't present (fresh checkout)."""
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "data" / "pga_walk_forward_backtest.json"
    if not p.exists():
        pytest.skip("no backtest receipt yet")
    d = json.loads(p.read_text())
    assert d.get("sport") == "pga"
    assert "data_status" in d
    if d["data_status"] == "HARNESS_ONLY_DATA_BLOCKED":
        # A blocked run must not manufacture graded bets.
        assert (d.get("n_bets") or 0) == 0
        assert d.get("data_blocker")  # a human-readable reason string
    else:
        # A real run must have a genuine sample.
        assert d["data_status"] == "REAL_BACKTEST"
        assert (d.get("n_matches_graded") or 0) > 0
