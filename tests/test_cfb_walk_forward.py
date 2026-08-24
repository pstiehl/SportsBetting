"""Tests for CFB Phase-1+2 walk-forward feature builder.

Covers:
  1. Leakage gate: snapshot for date D never contains games from date >= D
  2. Feature vector shape: always N_FEATURES elements
  3. Rolling efficiency: correct L5 calculation, None when < 3 prior games
  4. Bye week detection: 12-21 day gap flagged as bye
  5. Rest days capped at 14
  6. Conference tier diff: P5 vs G5 correct sign
  7. Smoke test: full walk-forward on synthetic games produces at least one fold
  8. Simulator: structure and edge gate
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.cfb_features.feature_builder import (
    FEATURE_NAMES,
    N_FEATURES,
    CFBGameRow,
    CFBTeamSnapshot,
    build_features,
    feature_vector,
    fit_rolling_snapshots,
    POWER_FIVE,
)
from flashcat.cfb_features.model import make_splits, walk_forward_evaluate, WalkForwardSplit
from flashcat.cfb_features.simulator import simulate


# ─────────────────────────── Helpers ──────────────────────────────────────


def _game(
    game_id: str,
    d: date,
    home: str,
    away: str,
    home_score: int = 35,
    away_score: int = 21,
    home_conf: str = "SEC",
    away_conf: str = "SEC",
    week: int = 1,
    season: int = 2022,
) -> CFBGameRow:
    return CFBGameRow(
        game_id=game_id,
        game_date=d,
        season=season,
        week=week,
        home=home,
        away=away,
        home_conference=home_conf,
        away_conference=away_conf,
        home_score=home_score,
        away_score=away_score,
        home_won=(home_score > away_score),
    )


def _synthetic_games(
    n_days: int = 90,
    games_per_day: int = 8,
    season: int = 2022,
    start_date: date = date(2022, 9, 1),
) -> list[CFBGameRow]:
    """Generate synthetic CFB games between 20 teams over n_days.

    CFB has fewer but larger-margin games than NBA/NFL.
    ``games_per_day`` games per day — real CFB has ~60 FBS games on Saturdays
    and 0-2 on weekdays.  We spread them evenly for harness testing.
    """
    teams = [
        "ALA", "UGA", "OSU", "MIC", "CLE", "LSU",
        "AUB", "FLA", "TEN", "ARK", "MIS", "SCR",
        "ORE", "USC", "UCL", "WAS", "STF", "CAL",
        "OKL", "TEX",
    ]
    # Mix P5 and G5 conferences
    confs = {
        "ALA": "SEC", "UGA": "SEC", "OSU": "Big Ten", "MIC": "Big Ten",
        "CLE": "ACC", "LSU": "SEC", "AUB": "SEC", "FLA": "SEC",
        "TEN": "SEC", "ARK": "SEC", "MIS": "SEC", "SCR": "Mountain West",
        "ORE": "Pac-12", "USC": "Pac-12", "UCL": "Pac-12", "WAS": "Pac-12",
        "STF": "Pac-12", "CAL": "Pac-12", "OKL": "Big 12", "TEX": "Big 12",
    }
    games: list[CFBGameRow] = []
    idx = 0
    for day in range(n_days):
        d = start_date + timedelta(days=day)
        used: set[str] = set()
        for k in range(games_per_day):
            home = teams[(idx + k * 2) % len(teams)]
            away = teams[(idx + k * 2 + 1) % len(teams)]
            if home == away or home in used or away in used:
                continue
            used.add(home)
            used.add(away)
            home_score = 20 + ((idx * 7 + k * 13) % 35)
            away_score = 14 + ((idx * 11 + k * 7) % 35)
            games.append(_game(
                game_id=f"syn_{day:03d}_{k:02d}",
                d=d,
                home=home,
                away=away,
                home_score=home_score,
                away_score=away_score,
                home_conf=confs.get(home, "Conference USA"),
                away_conf=confs.get(away, "Mountain West"),
                week=(day // 7) + 1,
                season=season,
            ))
            idx += 1
    return games


# ─────────────────────────── Leakage gate tests ───────────────────────────


def test_no_leakage_in_snapshots():
    """Snapshots must only contain games strictly before the snapshot date."""
    games = _synthetic_games(n_days=40)
    snapshots = fit_rolling_snapshots(games)
    for (team, snap_date), snap in snapshots.items():
        for g in snap.recent_games:
            assert g.game_date < snap_date, (
                f"LEAKAGE: snapshot for {team} on {snap_date} "
                f"contains game from {g.game_date}"
            )


def test_snapshot_contains_only_team_games():
    """Each team snapshot only holds games that team participated in."""
    games = _synthetic_games(n_days=30)
    snapshots = fit_rolling_snapshots(games)
    for (team, snap_date), snap in snapshots.items():
        for g in snap.recent_games:
            assert g.home == team or g.away == team, (
                f"Snapshot for {team} contains game where team is neither "
                f"home={g.home} nor away={g.away}"
            )


# ─────────────────────────── Feature names / shape ────────────────────────


def test_feature_names_count():
    assert len(FEATURE_NAMES) == N_FEATURES


def test_feature_vector_length():
    feat = {name: 1.0 for name in FEATURE_NAMES}
    vec = feature_vector(feat)
    assert vec is not None
    assert len(vec) == N_FEATURES


def test_feature_vector_none_imputed_to_zero():
    feat = {name: None for name in FEATURE_NAMES}
    feat["home_field_flag"] = 1.0
    feat["conf_tier_diff"] = 0.0
    vec = feature_vector(feat)
    assert vec is not None
    assert len(vec) == N_FEATURES
    idx_off = FEATURE_NAMES.index("off_eff_l5_diff")
    assert vec[idx_off] == 0.0


def test_feature_vector_fill_mean():
    feat = {name: None for name in FEATURE_NAMES}
    feat["home_field_flag"] = 1.0
    feat["conf_tier_diff"] = 0.0
    fill = {"off_eff_l5_diff": 7.5}
    vec = feature_vector(feat, fill_mean=fill)
    assert vec is not None
    idx = FEATURE_NAMES.index("off_eff_l5_diff")
    assert abs(vec[idx] - 7.5) < 1e-9


# ─────────────────────────── Rolling efficiency ───────────────────────────


def test_off_eff_l5_insufficient_history():
    """off_eff_l5 is None when team has < 3 prior games."""
    d = date(2022, 9, 1)
    # Only 2 prior games for ALA
    g0 = _game("g0", d, "ALA", "UGA", 35, 21)
    g1 = _game("g1", d + timedelta(days=7), "ALA", "LSU", 28, 14)
    snap = CFBTeamSnapshot(
        team="ALA",
        as_of=d + timedelta(days=14),
        recent_games=[g0, g1],
    )
    assert snap.off_eff_l5 is None, "Should be None with only 2 games"


def test_off_eff_l5_correct():
    """off_eff_l5 is correct average of last 5 scores."""
    team = "ALA"
    base = date(2022, 9, 1)
    pts = [35, 42, 28, 55, 21]
    prior_games: list[CFBGameRow] = []
    for i, p in enumerate(pts):
        g = _game(f"g{i}", base + timedelta(days=i * 7), team, "OPP",
                  home_score=p, away_score=10)
        prior_games.append(g)
    snap = CFBTeamSnapshot(
        team=team,
        as_of=base + timedelta(days=40),
        recent_games=prior_games,
    )
    expected = sum(pts) / len(pts)
    assert abs(snap.off_eff_l5 - expected) < 0.01, (
        f"Expected {expected}, got {snap.off_eff_l5}"
    )


# ─────────────────────────── Rest days & bye ──────────────────────────────


def test_rest_days_capped_at_14():
    d0 = date(2022, 9, 1)
    d1 = d0 + timedelta(days=30)  # 30 days rest
    g0 = _game("g0", d0, "ALA", "UGA")
    snap = CFBTeamSnapshot(
        team="ALA", as_of=d1, recent_games=[g0]
    )
    rd = snap.rest_days(d1)
    assert rd == 14.0, f"Rest days should be capped at 14, got {rd}"


def test_rest_days_normal():
    d0 = date(2022, 9, 1)
    d1 = d0 + timedelta(days=7)
    g0 = _game("g0", d0, "ALA", "UGA")
    snap = CFBTeamSnapshot(team="ALA", as_of=d1, recent_games=[g0])
    rd = snap.rest_days(d1)
    assert rd == 7.0, f"Expected 7.0, got {rd}"


def test_bye_week_detected():
    """A 14-day gap flags as bye."""
    d0 = date(2022, 9, 1)
    d1 = d0 + timedelta(days=14)  # 14 days = bye window
    g0 = _game("g0", d0, "ALA", "UGA")
    snap = CFBTeamSnapshot(team="ALA", as_of=d1, recent_games=[g0])
    assert snap.had_bye(d1), "14-day gap should flag as bye"


def test_normal_week_not_bye():
    """A 7-day gap is a normal week, not a bye."""
    d0 = date(2022, 9, 1)
    d1 = d0 + timedelta(days=7)
    g0 = _game("g0", d0, "ALA", "UGA")
    snap = CFBTeamSnapshot(team="ALA", as_of=d1, recent_games=[g0])
    assert not snap.had_bye(d1), "7-day gap should NOT flag as bye"


# ─────────────────────────── Conference tier diff ─────────────────────────


def test_conf_tier_diff_p5_vs_g5():
    from flashcat.cfb_features.feature_builder import _conf_tier_diff
    # P5 home vs G5 away → +1
    assert _conf_tier_diff("SEC", "Mountain West") == 1.0
    # G5 home vs P5 away → -1
    assert _conf_tier_diff("Mountain West", "SEC") == -1.0
    # Same tier → 0
    assert _conf_tier_diff("SEC", "Big Ten") == 0.0
    assert _conf_tier_diff("Mountain West", "Conference USA") == 0.0


# ─────────────────────────── Margin volatility ───────────────────────────


def test_margin_volatility_l5():
    """margin_volatility_l5 is the sample std dev of L5 margins from team's POV."""
    import math
    team = "ALA"
    base = date(2022, 9, 1)
    # ALA wins by 10, loses by 5, wins by 20, loses by 3, wins by 30
    margins = [10, -5, 20, -3, 30]
    prior_games: list[CFBGameRow] = []
    for i, m in enumerate(margins):
        if m > 0:
            g = _game(f"g{i}", base + timedelta(days=i * 7),
                      team, "OPP", home_score=100 + m, away_score=100)
        else:
            g = _game(f"g{i}", base + timedelta(days=i * 7),
                      team, "OPP", home_score=100, away_score=100 - m)
        prior_games.append(g)
    snap = CFBTeamSnapshot(team=team, as_of=base + timedelta(days=40), recent_games=prior_games)
    vol = snap.margin_volatility_l5
    assert vol is not None
    # Manual std dev check
    mean = sum(margins) / len(margins)
    variance = sum((x - mean) ** 2 for x in margins) / (len(margins) - 1)
    expected = math.sqrt(variance)
    assert abs(vol - expected) < 0.01, f"Expected vol {expected:.3f}, got {vol:.3f}"


# ─────────────────────────── Walk-forward smoke tests ─────────────────────


def test_walk_forward_smoke():
    """Full walk-forward on synthetic games should produce at least 1 fold.

    120 days × 8 games/day = ~960 synthetic games. With train_window=60d,
    warmup=20d, eval_window=10d: first eval window starts day 20,
    training covers days -40..19 → ~(40+20)×8 = ~480 examples.
    """
    games = _synthetic_games(n_days=120, games_per_day=8)
    snapshots = fit_rolling_snapshots(games)

    folds = walk_forward_evaluate(
        games,
        snapshots,
        train_window_days=60,
        eval_window_days=10,
        warmup_days=20,
    )
    assert len(folds) > 0, (
        "Walk-forward should produce at least 1 fold on 120d × 8 games/day"
    )


def test_walk_forward_no_leakage_in_splits():
    """Every fold must have train_end strictly before eval_start."""
    start = date(2022, 9, 1)
    end = date(2023, 1, 10)
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


def test_walk_forward_split_leakage_assertion():
    """WalkForwardSplit.__post_init__ should raise if train_end >= eval_start."""
    with pytest.raises(AssertionError, match="leakage"):
        WalkForwardSplit(
            train_start=date(2022, 9, 1),
            train_end=date(2022, 10, 15),
            eval_start=date(2022, 10, 15),  # same day = leakage
            eval_end=date(2022, 11, 14),
        )


# ─────────────────────────── Simulator tests ──────────────────────────────


def test_simulator_flat_stake():
    """Simulator returns correct structure."""
    games = _synthetic_games(n_days=120, games_per_day=8)
    snapshots = fit_rolling_snapshots(games)
    folds = walk_forward_evaluate(
        games, snapshots,
        train_window_days=60, eval_window_days=10, warmup_days=20,
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
        assert "data_notes" in summary


def test_simulator_edge_gate_impossible():
    """Edge gate of 1.0 (impossible) should produce 0 bets."""
    games = _synthetic_games(n_days=120, games_per_day=8)
    snapshots = fit_rolling_snapshots(games)
    folds = walk_forward_evaluate(
        games, snapshots,
        train_window_days=60, eval_window_days=10, warmup_days=20,
    )
    bets, summary = simulate(folds, edge_gate=1.0)
    assert summary["overall"]["n_bets"] == 0


def test_loss_buckets_sum_to_total_losses():
    """Loss bucket counts must sum to total number of losing bets."""
    games = _synthetic_games(n_days=120, games_per_day=8)
    snapshots = fit_rolling_snapshots(games)
    folds = walk_forward_evaluate(
        games, snapshots,
        train_window_days=60, eval_window_days=10, warmup_days=20,
    )
    bets, summary = simulate(folds)
    total_losses = sum(1 for b in bets if not b.won)
    bucket_sum = sum(summary["loss_buckets"]["counts"].values())
    assert bucket_sum == total_losses, (
        f"Loss bucket sum {bucket_sum} != total losses {total_losses}"
    )
