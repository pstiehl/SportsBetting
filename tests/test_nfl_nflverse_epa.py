"""Unit tests for the NFL nflfastR EPA connector."""

from __future__ import annotations

import math
from datetime import date

import pytest

from flashcat.sources import nfl_nflverse_epa as nfle
from flashcat.sources.nfl_nflverse_epa import (
    DEFAULT_COEFFS,
    NFLNflfastREPA,
    NFL_MARGIN_SIGMA,
    diff_to_home_prob,
    fit_ols_walk_forward,
    predicted_diff,
)


def test_predicted_diff_uses_coefficients():
    diff = predicted_diff(0.1, -0.05, -0.02, 0.05, is_home=True)
    assert isinstance(diff, float)
    assert diff > 0  # better off + better def = home favored


def test_diff_to_home_prob_calibrated_to_sigma_13_86():
    # +13.86 point diff should yield ~84% (1 sigma).
    p = diff_to_home_prob(NFL_MARGIN_SIGMA)
    assert 0.80 < p < 0.86


def test_diff_to_home_prob_zero_neutral():
    p = diff_to_home_prob(0.0)
    assert abs(p - 0.5) < 1e-6


def test_fit_ols_walk_forward_recovers_linear_relationship():
    import random
    random.seed(7)
    games = []
    true_alpha = 1.0
    true_beta_off = 50.0
    true_beta_def = -40.0
    true_beta_hfa = 2.5
    for _ in range(500):
        off_diff = random.gauss(0, 0.08)
        def_diff = random.gauss(0, 0.08)
        hfa = random.choice([0.0, 1.0])
        noise = random.gauss(0, 5)
        margin = (
            true_alpha
            + true_beta_off * off_diff
            + true_beta_def * def_diff
            + true_beta_hfa * hfa
            + noise
        )
        games.append({
            "off_epa_diff": off_diff,
            "def_epa_diff": def_diff,
            "hfa": hfa,
            "margin": margin,
        })
    fit = fit_ols_walk_forward(games)
    assert fit is not None
    # Tolerance is generous — 500 rows with σ=5 noise will wobble.
    assert abs(fit["beta_off"] - true_beta_off) < 8.0
    assert abs(fit["beta_def"] - true_beta_def) < 8.0
    assert abs(fit["beta_hfa"] - true_beta_hfa) < 1.5


def test_fit_returns_none_on_insufficient_data():
    assert fit_ols_walk_forward([]) is None
    assert fit_ols_walk_forward([{"off_epa_diff": 0, "def_epa_diff": 0, "hfa": 0, "margin": 0}]) is None


def test_connector_returns_empty_without_nfl_data_py(monkeypatch):
    """When nfl_data_py is unavailable the connector emits no events."""
    src = NFLNflfastREPA()
    # Force schedule load to fail like nfl_data_py is missing.
    monkeypatch.setattr(src, "_load_schedules", lambda seasons: None)
    out = src.fetch_events(date(2024, 9, 1), date(2024, 9, 30))
    assert out == []


def test_other_sports_return_nothing():
    src = NFLNflfastREPA()
    assert src.fetch_events(date(2024, 9, 1), date(2024, 9, 30), sport="mlb") == []


# --- PR #12: NFL Next Gen Stats CPOE connector ---------------------------------


def test_cpoe_to_home_prob_blends_with_epa():
    """Positive CPOE differential pulls home win prob up; EPA baseline is added."""
    from flashcat.sources.nfl_nflverse_epa import _cpoe_team_diff_to_home_prob

    # Neutral EPA baseline, +5pp CPOE differential → home > 0.5
    p = _cpoe_team_diff_to_home_prob(5.0, epa_pred_diff=0.0)
    assert p > 0.5
    # Negative CPOE differential → home < 0.5
    p = _cpoe_team_diff_to_home_prob(-5.0, epa_pred_diff=0.0)
    assert p < 0.5
    # Zero everything → 0.5
    p = _cpoe_team_diff_to_home_prob(0.0, epa_pred_diff=0.0)
    assert abs(p - 0.5) < 1e-6


def test_ngs_connector_handles_offline(monkeypatch):
    """Both NGS-direct and pbp-fallback unreachable → empty list, no raise."""
    from flashcat.sources.nfl_nflverse_epa import NFLNextGenCPOE

    src = NFLNextGenCPOE()
    monkeypatch.setattr(src, "_try_ngs_direct", lambda seasons: {})
    monkeypatch.setattr(src, "_team_cpoe_from_pbp", lambda seasons: {})
    out = src.fetch_events(date(2024, 9, 1), date(2024, 9, 30))
    assert out == []


def test_ngs_connector_other_sport():
    from flashcat.sources.nfl_nflverse_epa import NFLNextGenCPOE

    src = NFLNextGenCPOE()
    assert src.fetch_events(date(2024, 9, 1), date(2024, 9, 30), sport="mlb") == []


def test_ngs_connector_emits_events_with_pbp_fallback(monkeypatch):
    """When pbp CPOE rolls up cleanly + schedule loads, emit per-game probs."""
    from flashcat.sources.nfl_nflverse_epa import NFLNextGenCPOE

    src = NFLNextGenCPOE()
    monkeypatch.setattr(src, "_try_ngs_direct", lambda seasons: {})
    monkeypatch.setattr(src, "_team_cpoe_from_pbp", lambda seasons: {"KC": 4.5, "DET": -1.2})
    monkeypatch.setattr(src, "_load_schedules", lambda seasons: [
        {"date": date(2024, 9, 15), "home": "KC", "away": "DET"},
    ])
    monkeypatch.setattr(src, "_team_epa_baseline", lambda seasons: {})
    out = src.fetch_events(date(2024, 9, 1), date(2024, 9, 30))
    assert len(out) == 1
    ev = out[0]
    assert ev.sport == "nfl"
    assert ev.home == "KC"
    assert ev.away == "DET"
    # KC has higher CPOE → home prob > 0.5
    assert ev.source_probs[0].home_win_prob > 0.5
    assert ev.source_probs[0].source == "nfl-nextgen-cpoe"


def test_ngs_connector_is_marked_live():
    """The connector flips from stub to live so weights pick it up."""
    from flashcat.sources.nfl_nflverse_epa import NFLNextGenCPOE

    assert NFLNextGenCPOE.is_live is True
    assert NFLNextGenCPOE.version != "stub"
