"""Tests for the MLB Statcast lineup connector.

Pinned fixtures only — no live API calls. We monkey-patch the cached
HTTP helper to return fixture bytes.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from flashcat.sources import mlb_statcast_lineup as scl
from flashcat.sources.mlb_statcast_lineup import (
    DEFAULT_INTERCEPT,
    DEFAULT_SLOPE,
    LEAGUE_AVG_XWOBA,
    MLBStatcastLineup,
    PA_WEIGHTS_9,
    compute_team_offense_score,
    fit_calibration_from_backfill,
    score_diff_to_home_prob,
)


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def patched_cached_get(monkeypatch):
    """Route _cached_get to fixtures by URL substring."""
    schedule_bytes = (FIXTURES / "statsapi_schedule.json").read_bytes()
    savant_bytes = (FIXTURES / "savant_batter_sample.csv").read_bytes()

    def fake_get(url: str, cache_file: str, **_kwargs):
        if "statsapi.mlb.com" in url and "schedule" in url:
            return schedule_bytes
        if "baseballsavant.mlb.com" in url:
            return savant_bytes
        return None

    monkeypatch.setattr(scl, "_cached_get", fake_get)
    return fake_get


def test_pa_weights_sum_close_to_phil_spec():
    # Phil's spec: 13.5, 12.6, 11.8, 11.2, 10.6, 10.1, 9.6, 9.1, 8.6 (%)
    # which sums to 97.1% — the residual covers extra-inning PAs and is
    # renormalized inside compute_team_offense_score.
    assert abs(sum(PA_WEIGHTS_9) - 0.971) < 0.001
    # Weights are monotonically decreasing.
    assert all(
        PA_WEIGHTS_9[i] >= PA_WEIGHTS_9[i + 1] for i in range(len(PA_WEIGHTS_9) - 1)
    )


def test_compute_team_offense_score_uses_pa_weights():
    # Every batter has the same xwOBA → score should be xwoba × pitcher_allowed.
    score = compute_team_offense_score([0.350] * 9, 0.300)
    assert abs(score - 0.350 * 0.300) < 1e-9


def test_compute_team_offense_score_handles_missing_slots():
    score = compute_team_offense_score([], 0.310)
    assert abs(score - LEAGUE_AVG_XWOBA * 0.310) < 1e-9


def test_score_diff_to_home_prob_monotonic():
    p_neg = score_diff_to_home_prob(0.0, 0.02)
    p_zero = score_diff_to_home_prob(0.0, 0.0)
    p_pos = score_diff_to_home_prob(0.02, 0.0)
    assert p_neg < p_zero < p_pos
    assert 0.05 <= p_neg <= 0.95
    assert 0.05 <= p_pos <= 0.95


def test_default_calibration_loaded_without_file(monkeypatch, tmp_path):
    fake = tmp_path / "calibration.json"
    monkeypatch.setattr(scl, "CALIBRATION_PATH", fake)
    slope, intercept = scl._load_calibration(season=2024)
    assert slope == DEFAULT_SLOPE
    assert intercept == DEFAULT_INTERCEPT


def test_fetch_events_uses_fixture(patched_cached_get):
    src = MLBStatcastLineup()
    events = src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    assert len(events) == 1
    ev = events[0]
    assert ev.sport == "mlb"
    assert ev.home == "Colorado Rockies"
    assert ev.away == "Los Angeles Dodgers"
    assert len(ev.source_probs) == 1
    sp = ev.source_probs[0]
    assert sp.source == "mlb-statcast-lineup"
    assert 0.05 <= sp.home_win_prob <= 0.95
    # Notes should include the home/away offense scores.
    assert "home_off=" in sp.notes
    assert "diff=" in sp.notes


def test_other_sports_return_nothing(patched_cached_get):
    src = MLBStatcastLineup()
    assert src.fetch_events(date(2026, 5, 30), date(2026, 5, 30), sport="nfl") == []


def test_calibration_fit_recovers_known_slope():
    """Generate synthetic data with known slope and confirm we recover it."""
    import math
    import random
    random.seed(42)
    true_alpha = 0.05
    true_beta = 10.0

    def sig(x):
        return 1.0 / (1.0 + math.exp(-x))

    records = []
    for _ in range(2000):
        diff = random.gauss(0, 0.02)
        p = sig(true_alpha + true_beta * diff)
        y = random.random() < p
        records.append({"home_off": diff, "away_off": 0.0, "home_won": y})
    fit = fit_calibration_from_backfill(records)
    assert fit is not None
    alpha, beta = fit
    # Allow generous tolerance — 2000 samples can wobble.
    assert abs(beta - true_beta) < 4.0
    assert abs(alpha - true_alpha) < 0.3
