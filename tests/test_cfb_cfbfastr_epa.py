"""Unit tests for the CFB cfbfastR/PPA EPA connector."""

from __future__ import annotations

import math
from datetime import date

import pytest

from flashcat.sources.cfb_cfbfastr_epa import (
    CFB_MARGIN_SIGMA,
    DEFAULT_COEFFS,
    POWER_FIVE,
    CFBCfbfastREPA,
    _conf_dummy,
    diff_to_home_prob,
    fit_ols_walk_forward,
    predicted_diff,
)


def test_cfb_sigma_wider_than_nfl():
    """16.5 > 13.86 (NFL) — CFB margins are wider because of FBS talent gap."""
    assert CFB_MARGIN_SIGMA > 13.86
    assert CFB_MARGIN_SIGMA < 25.0  # but not silly-wide either


def test_predicted_diff_basic():
    diff = predicted_diff(0.2, -0.1, -0.05, 0.05, is_home=True)
    assert isinstance(diff, float)
    assert diff > 0  # better off + better def + HFA = home favored


def test_predicted_diff_includes_conference_dummy():
    """Conference dummy adds points when Power-5 hosts G5."""
    coeffs = DEFAULT_COEFFS
    no_conf = predicted_diff(0.0, 0.0, 0.0, 0.0, is_home=True, conf_dummy=0.0)
    p5_hosts_g5 = predicted_diff(0.0, 0.0, 0.0, 0.0, is_home=True, conf_dummy=1.0)
    g5_hosts_p5 = predicted_diff(0.0, 0.0, 0.0, 0.0, is_home=True, conf_dummy=-1.0)
    assert p5_hosts_g5 > no_conf > g5_hosts_p5
    assert (p5_hosts_g5 - no_conf) == pytest.approx(coeffs["beta_conf"])
    assert (no_conf - g5_hosts_p5) == pytest.approx(coeffs["beta_conf"])


def test_diff_to_home_prob_zero_neutral():
    assert abs(diff_to_home_prob(0.0) - 0.5) < 1e-6


def test_diff_to_home_prob_one_sigma_at_16_5():
    """+16.5 point diff ~ 1 sigma → ~84% win prob."""
    p = diff_to_home_prob(CFB_MARGIN_SIGMA)
    assert 0.80 < p < 0.86


def test_diff_to_home_prob_clipped():
    assert diff_to_home_prob(-500.0) == pytest.approx(0.03, abs=1e-6)
    assert diff_to_home_prob(500.0) == pytest.approx(0.97, abs=1e-6)


def test_conf_dummy_logic():
    assert _conf_dummy("SEC", "Sun Belt") == 1.0
    assert _conf_dummy("Sun Belt", "SEC") == -1.0
    assert _conf_dummy("SEC", "Big Ten") == 0.0  # both P5
    assert _conf_dummy("Sun Belt", "MAC") == 0.0  # both G5
    assert _conf_dummy(None, "SEC") == -1.0
    assert _conf_dummy("ACC", None) == 1.0


def test_power_five_set_complete():
    """All 5 P5 conferences are in the set."""
    assert {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12"} == POWER_FIVE


def test_fit_ols_walk_forward_recovers_relationship():
    import random
    random.seed(11)
    true_alpha = 0.5
    true_beta_off = 50.0
    true_beta_def = -40.0
    true_beta_hfa = 2.5
    true_beta_conf = 9.0
    games = []
    for _ in range(800):
        off = random.gauss(0, 0.08)
        deff = random.gauss(0, 0.08)
        hfa = random.choice([0.0, 1.0])
        conf = random.choice([-1.0, 0.0, 0.0, 1.0])
        noise = random.gauss(0, 6)
        margin = (
            true_alpha
            + true_beta_off * off
            + true_beta_def * deff
            + true_beta_hfa * hfa
            + true_beta_conf * conf
            + noise
        )
        games.append({
            "off_ppa_diff": off,
            "def_ppa_diff": deff,
            "hfa": hfa,
            "conf_dummy": conf,
            "margin": margin,
        })
    fit = fit_ols_walk_forward(games)
    assert fit is not None
    assert abs(fit["beta_off"] - true_beta_off) < 10.0
    assert abs(fit["beta_def"] - true_beta_def) < 10.0
    assert abs(fit["beta_hfa"] - true_beta_hfa) < 1.5
    assert abs(fit["beta_conf"] - true_beta_conf) < 2.0


def test_fit_returns_none_on_insufficient_data():
    assert fit_ols_walk_forward([]) is None
    games = [{
        "off_ppa_diff": 0, "def_ppa_diff": 0, "hfa": 0, "conf_dummy": 0, "margin": 0
    }]
    assert fit_ols_walk_forward(games) is None


def test_connector_returns_empty_outside_sport(monkeypatch):
    """Asking for sport='nba' yields no events."""
    src = CFBCfbfastREPA()
    assert src.fetch_events(date(2024, 9, 1), date(2024, 9, 7), sport="nba") == []  # type: ignore[arg-type]


def test_connector_returns_empty_when_api_unreachable(monkeypatch):
    """Network failure → connector returns [] (graceful degradation)."""
    from flashcat.sources import cfb_cfbfastr_epa as mod

    def _raise(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "_cfbd_get", _raise)
    src = CFBCfbfastREPA()
    # Use far-future dates to bypass any 24h cache files.
    out = src.fetch_events(date(2099, 9, 1), date(2099, 9, 7), sport="cfb")
    assert out == []


def test_load_results_returns_historical_when_schedule_available(monkeypatch, tmp_path):
    from flashcat.sources import cfb_cfbfastr_epa as mod

    fake_games = [
        {
            "start_date": "2024-09-07T18:00:00Z",
            "home_team": "Alabama",
            "away_team": "South Florida",
            "home_conference": "SEC",
            "away_conference": "American",
            "home_points": 42,
            "away_points": 16,
            "week": 2,
            "season": 2024,
        }
    ]

    def _fake_get(path, params, timeout=30.0):
        if path == "games":
            return fake_games
        return []

    monkeypatch.setattr(mod, "_cfbd_get", _fake_get)
    monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)

    src = CFBCfbfastREPA()
    res = src.load_results(date(2024, 9, 1), date(2024, 9, 30))
    assert len(res) == 1
    r = res[0]
    assert r.sport == "cfb"
    assert r.home == "Alabama"
    assert r.away == "South Florida"
    assert r.home_won is True
    assert r.home_score == 42
    assert r.away_score == 16
