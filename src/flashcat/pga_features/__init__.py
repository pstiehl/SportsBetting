"""PGA (golf) feature-expansion harness — Week 7 of the rotation.

A near-mechanical PORT of ``src/flashcat/wta_features/`` (Week 6) to PGA Tour
head-to-head matchups: pure-function feature builders, leakage-gated rolling
form, a sport-agnostic walk-forward logistic harness (reused from
``nfl_features.model``), and a flat-$100 simulator with golf-specific loss
buckets.

This is OFF-CI, opt-in R&D — see ``docs/FEATURE_EXPANSION_PLAYBOOK.md``.

Event shape
-----------
PGA H2H matchups are the same shape as tennis singles: two players, the
bettor picks which one finishes ahead over the tournament. The production
``sources/pga_datagolf.py`` connector already synthesizes these Events with
``home_win_prob`` = Bradley-Terry on DataGolf skill. ``home`` is the
alphabetically-first player; ``home_won`` is True when that player finished
strictly ahead (ties => push => dropped).

Data spine (when available): DataGolf pre-tournament win% / skill (skill
prior + BT matchup prior via ``datagolf-sg``) and a matchup closing
moneyline (``market-close``, also the CLV proxy), both read from
``data/source_history.db`` where a backfill persisted them.

Sport-mapping vs the WTA catalog
--------------------------------
    tennis surface        -> PGA course difficulty tier (easy/standard/hard/major)
    tennis ranking points -> DataGolf pre-tournament win% / skill estimate
    tennis rank-BT prior  -> datagolf-sg Bradley-Terry matchup prior
    tennis games/sets won -> made-cut rate + exponential finish-quality score

Phase-1 feature catalog (12 features):
  Priors / market (3)
    market_prob_home           — devigged closing matchup moneyline (CLV proxy)
    skill_bt_prob_home         — DataGolf Bradley-Terry matchup prior
    skill_bt_minus_market_pp   — BT prior minus market (calibration delta)
  Skill / rating (2)
    win_pct_log_ratio          — log(home pre-tournament win% / away win%)
    skill_diff                 — home SG-total skill minus away skill
  Rolling form (4)
    h2h_form_l10_diff          — rolling finish-ahead rate over last 10 starts
    finish_quality_l10_diff    — avg exponential finish-quality over last 10 starts
    made_cut_pct_l10_diff      — made-cut rate over last 10 starts
    course_tier_quality_l10_diff — finish quality on this course-difficulty tier
  Schedule / fatigue (2)
    rest_days_diff             — days since previous start (home - away), capped 90
    starts_l28_diff            — starts in trailing 28 days (home - away)
  Head-to-head (1)
    h2h_home_share             — home's historical H2H finish-ahead share (0.5 prior)

Dropped from the WTA catalog (documented in the PR):
  * sets_won_pct_l10_diff / games_won_pct — no golf analog. Replaced by
    made-cut rate + finish-quality; net catalog is 12 features (WTA's 13
    minus the two set/game features, plus made-cut + skill_diff).

CRITICAL DATA STATUS (Week 7, 2026-07-20): there are NO PGA rows in
source_history.db yet, and the only known source of genuine historical
closing MATCHUP odds — DataGolf's matchup-odds archive — is a PAID-tier
endpoint (OFF LIMITS per Phil's constraint) requiring an unset
``DATAGOLF_API_KEY``. No free source pairing player finishes with closing
matchup probabilities was found. So this Phase-1 harness ships leakage-gated
and unit-tested against a fixture, ready to run the moment a data source or
key is provided. The real walk-forward backtest is BLOCKED on data access,
NOT run against fabricated outcomes. See the PR body and playbook §6.
"""

from .feature_builder import (  # noqa: F401
    MatchupRow,
    FEATURE_NAMES,
    build_features,
    feature_vector,
    fit_rolling_rates,
    attach_priors_from_db,
    compute_h2h,
    event_id,
)

__all__ = [
    "MatchupRow",
    "FEATURE_NAMES",
    "build_features",
    "feature_vector",
    "fit_rolling_rates",
    "attach_priors_from_db",
    "compute_h2h",
    "event_id",
]
