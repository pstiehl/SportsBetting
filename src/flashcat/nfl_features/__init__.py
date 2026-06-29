"""NFL feature-expansion harness — Week 4 of the weekly sport rotation.

Mirrors ``src/flashcat/mlb_features/`` in spirit: pure-function feature
builders, leakage-gated rolling rates, a sport-agnostic walk-forward
logistic harness, and a flat-$100 simulator with NFL-specific loss
buckets.

This is OFF-CI, opt-in R&D — see ``docs/FEATURE_EXPANSION_PLAYBOOK.md``.

Phase-1 feature catalog (17 features):
  Priors (3)
    elo_prob_home          — 538 NFL Elo pre-game prob (web.archive 2023 snapshot)
    epa_prob_home          — nflfastR EPA model prob (existing connector)
    market_prob_home       — closing-moneyline implied prob (devigged)
  Prior-stack signals (3)
    elo_minus_market_pp    — 538 Elo minus market (calibration delta)
    epa_minus_market_pp    — EPA minus market
    priors_avg             — mean of (elo, epa, market) — composite prior
  Rolling form (6)
    off_epa_l4_diff        — home off-EPA/play L4 - away off-EPA/play L4
    def_epa_l4_diff        — home def-EPA/play L4 (allowed) - away (allowed)
    success_rate_l4_diff   — home success-rate L4 - away success-rate L4
    off_epa_l8_diff        — same as L4 but longer window
    pass_epa_l4_diff       — pass-only EPA/play L4
    rush_epa_l4_diff       — rush-only EPA/play L4
  Schedule / rest (5)
    rest_diff              — home_rest_days - away_rest_days
    home_off_bye           — home team coming off a bye (1.0 / 0.0)
    away_off_bye           — away team coming off a bye
    divisional             — divisional matchup (1.0 / 0.0)
    week_number            — regular-season week (normalized 1-18)
"""

from .feature_builder import (  # noqa: F401
    GameRow,
    FEATURE_NAMES,
    build_features,
    feature_vector,
    fit_rolling_rates,
    load_games_from_schedules,
    load_pbp_rollups,
)

__all__ = [
    "GameRow",
    "FEATURE_NAMES",
    "build_features",
    "feature_vector",
    "fit_rolling_rates",
    "load_games_from_schedules",
    "load_pbp_rollups",
]
