"""NBA expanded feature engineering for the walk-forward backtest — Phase 2.

Feature catalog (Phase 2 — targeting line_moved_against loss bucket):

  Rolling point-differential (L5 games)
    pt_diff_l5_home         rolling pt diff per game, home team, last 5
    pt_diff_l5_away         rolling pt diff per game, away team, last 5
    pt_diff_l5_diff         home − away (primary model signal)

  Rolling win percentage (L10 games)
    win_pct_l10_home        win % home team, last 10 games
    win_pct_l10_away        win % away team, last 10 games
    win_pct_l10_diff        home − away

  Schedule fatigue
    b2b_home                1 if home team played yesterday
    b2b_away                1 if away team played yesterday
    b2b_diff                b2b_away − b2b_home (positive = away disadvantaged)
    rest_days_diff          (home rest days) − (away rest days); capped ±7

  Season-level strength prior
    srs_diff                pre-existing bref SRS home − away (from source_history.db)

  Structural
    home_court_flag         constant 1.0 for HCA interpretability

Each builder is pure-function (takes raw game-log rows, returns features) so the
walk-forward harness can compute features as-of any cutoff without leakage.
The leakage gate is asserted in tests: ``asof < game_date`` always.
"""

from .feature_builder import (
    NBAGameRow,
    NBATeamSnapshot,
    FEATURE_NAMES,
    load_nba_game_logs,
    fit_rolling_snapshots,
    build_features,
    feature_vector,
    load_srs_from_db,
)

__all__ = [
    "NBAGameRow",
    "NBATeamSnapshot",
    "FEATURE_NAMES",
    "load_nba_game_logs",
    "fit_rolling_snapshots",
    "build_features",
    "feature_vector",
    "load_srs_from_db",
]
