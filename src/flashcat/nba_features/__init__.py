"""NBA expanded feature engineering for the walk-forward backtest.

Feature catalog (Phase 1 — what's actually backfillable from
``data/source_history.db`` without new scrapers):

  External priors (each devig'd, already in DB)
    raptor_prob_home              538 RAPTOR pre-game home win prob
    elo_modern_prob_home          538 modern-Elo pre-game home win prob
    bref_srs_prob_home            basketball-reference SRS+pace prob

  Derived prior-stack signals
    prior_consensus               mean of the priors available for this game
    prior_dispersion              max - min across available priors (uncertainty)
    raptor_vs_elo_disagree        raptor - elo when both present (else 0)

  Rolling team form (derived from prior outcomes — no scrape needed)
    win_pct_l5_home, win_pct_l5_away
    win_pct_l10_home, win_pct_l10_away
    win_pct_diff_l10              home - away
    avg_margin_l10_home, avg_margin_l10_away  (margin proxy = log-odds of
        the consensus prior on each game; outcome-anchored not available
        because we don't have NBA box scores in this DB)

  Schedule / rest signals
    days_rest_home, days_rest_away
    days_rest_diff                home - away
    b2b_home, b2b_away            binary (rest == 1 day)

  Home-court
    is_home_back_in_arena         always 1 in this scope (placeholder)

Phase 2 candidates (explicitly OUT of scope for the walk-forward harness
in this PR; documented for the next rotation):

  - actual box-score rolling rates (off rating, def rating, pace) via
    basketball-reference per-game scrapes
  - top-scorer usage diff vs opponent defensive rating
  - altitude_diff (Denver/Utah)
  - injury/load-management report (balldontlie.io or RotoWire)
  - real market-close odds via The Odds API historical endpoint

This module is offline-by-default and pure: every loader reads from
``data/source_history.db`` and every feature is computable using only
events strictly before the target game's date. The leakage gate is
asserted in tests.
"""

from .feature_builder import (
    GameRow,
    FEATURE_NAMES,
    build_features,
    feature_vector,
    load_nba_games_from_history,
    normalize_team,
)

__all__ = [
    "GameRow",
    "FEATURE_NAMES",
    "build_features",
    "feature_vector",
    "load_nba_games_from_history",
    "normalize_team",
]
