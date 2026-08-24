"""CFB expanded feature engineering for the walk-forward backtest — Phase 1+2.

Phase-1 and Phase-2 implemented together (Phase-1 was lost in a workspace
wipe; this is a combined re-run, same methodology, same honest evaluation).

Feature catalog:

  Rolling team efficiency (L3/L5/L10 games)
    off_eff_l5_home         avg points scored per game, home team, last 5
    off_eff_l5_away         avg points scored per game, away team, last 5
    off_eff_l5_diff         home − away (offensive efficiency signal)
    def_eff_l5_home         avg points allowed per game, home team, last 5
    def_eff_l5_away         avg points allowed per game, away team, last 5
    def_eff_l5_diff         home − away (lower = better defense)
    net_eff_l5_diff         (off_eff_l5_diff) − (def_eff_l5_diff) combined

  Schedule fatigue
    rest_days_home          days since last game, home team (capped 14)
    rest_days_away          days since last game, away team (capped 14)
    rest_days_diff          (home rest) − (away rest)
    bye_home                1 if home team on bye last week
    bye_away                1 if away team on bye last week

  Turnover differential (L5 games — proxied from point margin variance)
    margin_volatility_home  std dev of point margins, home team L5
    margin_volatility_away  std dev of point margins, away team L5

  Conference and home field
    conf_tier_diff          +1 P5 home vs G5 away, -1 reversed, 0 same tier
    home_field_flag         constant 1.0 (isolates HFA in logistic coeff)

  Market context
    opening_prob_home       home-win probability from EPA connector (proxy for opening line)

Each builder is pure-function (takes raw game-log rows, returns features)
so the walk-forward harness computes features as-of any cutoff without leakage.
Leakage gate asserted in tests: asof < game_date always.

Data source: ESPN public scoreboard via CFBCfbfastREPA._load_schedule_from_espn()
(college football data API gated without paid API key; ESPN fallback is free).
2022–2024 seasons available; ~900 FBS games/season → ~2700 games over 3 seasons.
"""

from .feature_builder import (
    CFBGameRow,
    CFBTeamSnapshot,
    FEATURE_NAMES,
    load_cfb_game_logs,
    fit_rolling_snapshots,
    build_features,
    feature_vector,
)

__all__ = [
    "CFBGameRow",
    "CFBTeamSnapshot",
    "FEATURE_NAMES",
    "load_cfb_game_logs",
    "fit_rolling_snapshots",
    "build_features",
    "feature_vector",
]
