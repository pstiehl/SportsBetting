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
