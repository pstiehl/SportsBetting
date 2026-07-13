"""WTA (women's tennis) feature-expansion harness — Week 6 of the rotation.

A near-mechanical PORT of ``src/flashcat/atp_features/`` (Week 5) to the WTA
women's tour: pure-function feature builders, leakage-gated rolling rates, a
sport-agnostic walk-forward logistic harness (reused from ``nfl_features.model``),
and a flat-$100 simulator with tennis-specific loss buckets.

This is OFF-CI, opt-in R&D — see ``docs/FEATURE_EXPANSION_PLAYBOOK.md``.

Data spine: tennis-data.co.uk season xlsx (results, WRank/LRank, WPts/LPts,
surface, round, series, per-set scores, closing odds — Pinnacle preferred).
The xlsx is downloaded/cached by ``scripts/backfill_tennis_historical.py``.
Priors (tennis-rank-bt Bradley-Terry, market-close devigged) are read from
``data/source_history.db`` where the backfill persisted them.

WTA-specific difference vs the ATP catalog
------------------------------------------
The WTA main tour is **best-of-3 ONLY** — there are no best-of-5 Grand Slam
matches on the women's tour. The ATP ``best_of_5`` feature would be a
constant 0.0 across the entire WTA dataset and add exactly zero signal, so it
is **DROPPED** here: the Phase-1 WTA catalog is **13 features** (vs 14 for
ATP). Everything else in the ATP catalog ports directly.

Phase-1 feature catalog (13 features):
  Priors / market (3)
    market_prob_home        — devigged closing two-way moneyline (Pinnacle>B365>Avg)
    rank_bt_prob_home       — Bradley-Terry on WTA rank points (from source_history.db)
    rank_bt_minus_market_pp — BT prior minus market (calibration delta)
  Ranking (2)
    rank_log_ratio          — log(away_rank / home_rank) — lower rank number = better
    rank_points_log_ratio   — log(home_pts / away_pts)
  Rolling form (5)
    win_pct_l10_diff        — win% over last 10 matches (home - away)
    win_pct_l25_diff        — win% over last 25 matches
    surface_win_pct_l20_diff— surface-specific win% over last 20 same-surface matches
    games_won_pct_l10_diff  — share of games won across last 10 matches (serve/return proxy)
    sets_won_pct_l10_diff   — share of sets won across last 10 matches
  Schedule / fatigue (2)
    rest_days_diff          — days since previous match (home - away), capped
    matches_l14_diff        — matches played in trailing 14 days (home - away)
  Head-to-head (1)
    h2h_home_share          — home's historical H2H win share vs this opponent (0.5 prior)

Dropped from the ATP catalog (documented in the PR):
  * best_of_5 — WTA is best-of-3 only; the feature is constant 0.0 and adds
    zero signal on the women's tour.

Skipped in Phase 1 (same as ATP, documented in the PR):
  * sackmann-wta-elo surface-adjusted Elo — the JeffSackmann/tennis_wta
    GitHub repo returned 404/429 during the backfill, so the Elo source
    never populated. Highest-value Phase-2 lever.
  * serve hold % / return break % — tennis-data.co.uk does not publish
    per-match serve/return point stats; games-won share is our proxy.
  * real line-movement — only closing odds are archived, no opener.
"""

from .feature_builder import (  # noqa: F401
    MatchRow,
    FEATURE_NAMES,
    build_features,
    feature_vector,
    fit_rolling_rates,
    load_matches_from_cache,
    attach_priors_from_db,
)

__all__ = [
    "MatchRow",
    "FEATURE_NAMES",
    "build_features",
    "feature_vector",
    "fit_rolling_rates",
    "load_matches_from_cache",
    "attach_priors_from_db",
]
