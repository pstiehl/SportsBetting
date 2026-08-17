"""Tests for NBA Phase-2 walk-forward feature builder.

Covers:
  1. Leakage gate: snapshot for date D never contains games from date >= D
  2. Feature vector shape: always N_FEATURES elements
  3. B2B detection: correctly flags back-to-back games
  4. Rolling pt-diff: correct window (L5), zero-padded for insufficient history
  5. Smoke test: full walk-forward on 50 synthetic games produces at least one fold
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.nba_features.feature_builder import (
    FEATURE_NAMES,
    N_FEATURES,
    NBAGameRow,
    NBATeamSnapshot,
    build_features,
    feature_vector,
    fit_rolling_snapshots,
)
from flashcat.nba_features.model import make_splits, walk_forward_evaluate
from flashcat.nba_features.simulator import simulate


# ─────────────────────────── Helpers ──────────────────────────────────────


def _game(
    game_id: str,
    d: date,
    home: str,
    away: str,
    home_score: int = 110,
    away_score: int = 100,
) -> NBAGameRow:
    return NBAGameRow(
        game_id=game_id,
        game_date=d,
        season="2022-23",
        home=home,
        away=away,
        home_score=home_score,
        away_score=away_score,
        home_won=(home_score > away_score),
    )


def _synthetic_games(n_days: int = 60, games_per_day: int = 6) -> list[NBAGameRow]:
    """Generate synthetic games between 12 teams over n_days.

    ``games_per_day`` games are generated per day so that training windows
    accumulate enough examples (real NBA has ~12 games/night; 6 is enough
    to get 50+ examples in a 10-day window).
    """
    teams = ["BOS", "LAL", "GSW", "CHI", "MIA", "DEN",
             "PHX", "BKN", "MIL", "PHI", "DAL", "NOP"]
    games: list[NBAGameRow] = []
    start = date(2022, 10, 18)
    idx = 0
    for day in range(n_days):
        d = start + timedelta(days=day)
        used_teams: set[str] = set()
        for k in range(games_per_day):
            home = teams[(idx + k * 2) % len(teams)]
            away = teams[(idx + k * 2 + 1) % len(teams)]
            if home == away or home in used_teams or away in used_teams:
                continue
            used_teams.add(home)
            used_teams.add(away)
            # Vary scores so both home wins and away wins occur
            home_score = 100 + ((idx * 3 + k * 7) % 30)
            away_score = 100 + ((idx * 5 + k * 11) % 30)
            g = _game(
                game_id=f"test_{day:03d}_{k:02d}",
                d=d,
                home=home,
                away=away,
                home_score=home_score,
                away_score=away_score,
            )
            games.append(g)
            idx += 1
    return games


# ─────────────────────────── Leakage gate tests ───────────────────────────


def test_no_leakage_in_snapshots():
    """Every game in a snapshot must be strictly before the snapshot date."""
    games = _synthetic_games(n_days=40)
    snapshots = fit_rolling_snapshots(games)
    for (team, snap_date), snap in snapshots.items():
        for g in snap.recent_games:
            assert g.game_date < snap_date, (
                f"LEAKAGE: snapshot for {team} on {snap_date} "
                f"contains game from {g.game_date}"
            )


def test_snapshot_is_team_only():
    """Snapshot recent_games should only contain games the team played."""
    games = _synthetic_games(n_days=30)
    snapshots = fit_rolling_snapshots(games)
    for (team, snap_date), snap in snapshots.items():
        for g in snap.recent_games:
            assert g.home == team or g.away == team, (
                f"Snapshot for {team} contains game {g.game_id} "
                f"where team neither home={g.home} nor away={g.away}"
            )


# ─────────────────────────── Feature shape ────────────────────────────────


def test_feature_names_count():
    """FEATURE_NAMES must have exactly N_FEATURES entries."""
    assert len(FEATURE_NAMES) == N_FEATURES


def test_feature_vector_length():
    """feature_vector must always return a list of exactly N_FEATURES floats."""
    # Build a minimal feature dict with all non-None values
    feat: dict = {name: 0.0 for name in FEATURE_NAMES}
    vec = feature_vector(feat)
    assert vec is not None
    assert len(vec) == N_FEATURES


def test_feature_vector_none_fill():
    """None values are imputed to 0.0 by default."""
    feat = {name: None for name in FEATURE_NAMES}
    # Required features (b2b_diff, home_court_flag) must be set
    feat["b2b_home"] = 0.0
    feat["b2b_away"] = 0.0
    feat["b2b_diff"] = 0.0
    feat["home_court_flag"] = 1.0
    vec = feature_vector(feat)
    assert vec is not None
    assert len(vec) == N_FEATURES
    # Null-filled features default to 0.0
    idx_srs = FEATURE_NAMES.index("srs_diff")
    assert vec[idx_srs] == 0.0


def test_feature_vector_fill_mean():
    """fill_mean replaces None with specified mean."""
    feat = {name: None for name in FEATURE_NAMES}
    feat["b2b_home"] = 0.0
    feat["b2b_away"] = 0.0
    feat["b2b_diff"] = 0.0
    feat["home_court_flag"] = 1.0
    fill = {"srs_diff": 3.5, "rest_days_diff": 1.0}
    vec = feature_vector(feat, fill_mean=fill)
    assert vec is not None
    idx_srs = FEATURE_NAMES.index("srs_diff")
    assert abs(vec[idx_srs] - 3.5) < 1e-9


# ─────────────────────────── B2B detection ────────────────────────────────


def test_b2b_detection():
    """Team that plays on consecutive days is flagged as B2B on the second day."""
    d0 = date(2022, 11, 1)
    d1 = date(2022, 11, 2)
    g0 = _game("g0", d0, "BOS", "LAL", 110, 100)
    g1 = _game("g1", d1, "CHI", "BOS", 108, 105)  # BOS away on d1 (B2B)

    games = [g0, g1]
    snapshots = fit_rolling_snapshots(games)

    snap_bos_d1 = snapshots.get(("BOS", d1))
    assert snap_bos_d1 is not None
    assert snap_bos_d1.is_b2b(d1), "BOS should be on a back-to-back on d1"

    snap_chi_d1 = snapshots.get(("CHI", d1))
    assert snap_chi_d1 is not None
    assert not snap_chi_d1.is_b2b(d1), "CHI should NOT be on a back-to-back on d1"


def test_not_b2b_after_two_days_rest():
    """Team with 2 days rest is NOT flagged as B2B."""
    d0 = date(2022, 11, 1)
    d2 = date(2022, 11, 3)  # 2 days later
    g0 = _game("g0", d0, "BOS", "LAL", 110, 100)
    g1 = _game("g1", d2, "BOS", "CHI", 105, 102)

    games = [g0, g1]
    snapshots = fit_rolling_snapshots(games)

    snap_bos = snapshots.get(("BOS", d2))
    assert snap_bos is not None
    assert not snap_bos.is_b2b(d2), "2-day rest should not flag B2B"


# ─────────────────────────── Rolling pt diff ──────────────────────────────


def test_pt_diff_l5_insufficient_history():
    """pt_diff_l5 returns None when team has fewer than 3 prior games."""
    d = date(2022, 10, 18)
    g0 = _game("g0", d, "BOS", "LAL", 110, 100)
    # BOS has 1 game before d+1
    snap = NBATeamSnapshot(team="BOS", as_of=d + timedelta(days=1), recent_games=[g0])
    assert snap.pt_diff_l5 is None, "Should be None with only 1 game"


def test_pt_diff_l5_correct_value():
    """pt_diff_l5 is the avg pt diff from team perspective over last 5 games."""
    team = "BOS"
    games_before: list[NBAGameRow] = []
    start = date(2022, 10, 18)
    # BOS wins by +10, +12, +8, +15, +5 (home games) over 5 consecutive days
    diffs = [10, 12, 8, 15, 5]
    for i, diff in enumerate(diffs):
        g = _game(f"g{i}", start + timedelta(days=i), "BOS", "LAL",
                  home_score=100 + diff, away_score=100)
        games_before.append(g)

    snap = NBATeamSnapshot(team=team, as_of=start + timedelta(days=5), recent_games=games_before)
    expected = sum(diffs) / len(diffs)
    assert abs(snap.pt_diff_l5 - expected) < 0.01


# ─────────────────────────── Walk-forward smoke test ─────────────────────


def test_walk_forward_smoke():
    """Full walk-forward on synthetic games should produce at least 1 fold.

    Generates 120 days × 6 games/day ≈ 720 synthetic games. With
    train_window=30 days and warmup=20 days, the first evaluation window
    starts at day-20 and the training window covers days -10..19, giving
    ~120 games (6/day × 20 days) — well above the 50-example threshold.
    """
    games = _synthetic_games(n_days=120, games_per_day=6)
    snapshots = fit_rolling_snapshots(games)
    srs_lookup: dict = {}  # no SRS data — tests SRS=None path

    folds = walk_forward_evaluate(
        games, snapshots, srs_lookup,
        train_window_days=30,
        eval_window_days=10,
        warmup_days=20,
    )
    assert len(folds) > 0, "Walk-forward should produce at least 1 fold on 120d × 6 games/day"


def test_walk_forward_no_leakage_in_splits():
    """Every fold split must satisfy train_end < eval_start."""
    from flashcat.nba_features.model import WalkForwardSplit

    start = date(2022, 10, 18)
    end = date(2023, 4, 9)
    splits = make_splits(
        start, end,
        train_window_days=90,
        eval_window_days=30,
        warmup_days=30,
    )
    for s in splits:
        assert s.train_end < s.eval_start, (
            f"Leakage: train_end={s.train_end} >= eval_start={s.eval_start}"
        )


def test_simulator_flat_stake():
    """Simulator returns correct n_bets and structure."""
    games = _synthetic_games(n_days=120, games_per_day=6)
    snapshots = fit_rolling_snapshots(games)
    srs_lookup: dict = {}

    folds = walk_forward_evaluate(
        games, snapshots, srs_lookup,
        train_window_days=30,
        eval_window_days=10,
        warmup_days=20,
    )
    bets, summary = simulate(folds)
    o = summary["overall"]
    assert isinstance(o["n_bets"], int)
    assert o["n_bets"] >= 0
    if o["n_bets"] > 0:
        assert 0.0 <= o["win_rate"] <= 1.0
        assert isinstance(o["roi"], float)
        assert "per_year" in summary
        assert "loss_buckets" in summary


def test_simulator_edge_gate():
    """Edge gate of 1.0 (impossible) should produce 0 bets."""
    games = _synthetic_games(n_days=120, games_per_day=6)
    snapshots = fit_rolling_snapshots(games)
    srs_lookup: dict = {}
    folds = walk_forward_evaluate(
        games, snapshots, srs_lookup,
        train_window_days=30, eval_window_days=10, warmup_days=20,
    )
    bets, summary = simulate(folds, edge_gate=1.0)
    assert summary["overall"]["n_bets"] == 0
