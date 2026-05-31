"""MLB expanded feature engineering for the walk-forward backtest.

Feature catalog (Phase 1 — what Phil asked for):

  Team-offense rolling rates (L3 / L5 / L10 / L20)
    rs_l3, rs_l5, rs_l10, rs_l20    runs scored
    ra_l3, ra_l5, ra_l10, ra_l20    runs allowed
    rd_l10                          run differential L10
    win_pct_l10                     win % L10

  Starting pitcher (538-derived, season-snapshot at game time)
    pitcher_rating_diff             home rating - away rating (538 pitcher_rgs)
    pitcher_adj_diff                home adj - away adj
    pitcher_days_rest_home/away     days since previous start
    pitcher_days_rest_diff          home_rest - away_rest

  538 Elo (already in cache)
    elo_prob_home, rating_prob_home, elo_diff

  Park + weather (offline; from data/mlb_parks.json + Open-Meteo)
    park_runs_per_game              static park factor
    temp_f, wind_mph_to_cf          weather (optional; nullable)

  Umpire (placeholder for v1; per-umpire mean K%/BB% from Retrosheet)
    ump_k_pct, ump_bb_pct           (nullable; v2 source)

Each builder is pure-function (takes raw game-log rows, returns features) so the
walk-forward harness can compute features as-of any cutoff without leakage.
The leakage gate is asserted in tests: ``asof < game_date`` always.

This module is offline-by-default. The expensive live pulls (Statcast batter
exit velocity, ump strike-zone data, weather archive) are gated on network
availability and degrade to ``None`` features rather than crashing.
"""

from .feature_builder import (
    GameRow,
    RollingTeamFeatures,
    build_features,
    fit_rolling_rates,
    load_538_mlb_games,
    load_retrosheet_games,
)

__all__ = [
    "GameRow",
    "RollingTeamFeatures",
    "build_features",
    "fit_rolling_rates",
    "load_538_mlb_games",
    "load_retrosheet_games",
]
