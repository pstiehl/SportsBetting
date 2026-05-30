"""Tests for the MLB park-adjusted weather connector."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from flashcat.sources import mlb_statcast_lineup as scl
from flashcat.sources import mlb_weather as mw
from flashcat.sources.mlb_weather import (
    MLBWeather,
    humidity_multiplier,
    load_parks,
    pythagorean_win_prob,
    runs_expected,
    temperature_multiplier,
    wind_multiplier,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_parks_includes_known_venues():
    parks = load_parks()
    assert "Coors Field" in parks
    assert parks["Coors Field"]["dome"] is False
    assert parks["Tropicana Field"]["dome"] is True


def test_temperature_multiplier_warm_air_adds_runs():
    assert temperature_multiplier(80) > 1.0
    assert temperature_multiplier(60) < 1.0
    assert temperature_multiplier(70) == 1.0


def test_wind_multiplier_in_to_cf_subtracts():
    # park orientation = 0 (home plate → north). Wind blowing FROM south
    # (wind_direction=180) means wind blows toward north (toward CF) →
    # tailwind, adds runs.
    tail = wind_multiplier(15, 180, 0)
    assert tail > 1.0
    # Wind from north (toward home plate) → headwind, subtracts runs.
    head = wind_multiplier(15, 0, 0)
    assert head < 1.0


def test_wind_multiplier_below_threshold_neutral():
    assert wind_multiplier(5, 180, 0) == 1.0


def test_humidity_multiplier_only_affects_coors():
    assert humidity_multiplier(85, "Fenway Park") == 1.0
    assert humidity_multiplier(85, "Coors Field") < 1.0
    assert humidity_multiplier(50, "Coors Field") == 1.0


def test_runs_expected_dome_skips_weather():
    parks = load_parks()
    h, a = runs_expected(parks["Tropicana Field"], {"temperature_f": 95}, park_name="Tropicana Field")
    base = 4.5 * parks["Tropicana Field"]["run_env_baseline"]
    assert abs(h - base) < 1e-9
    assert abs(a - base) < 1e-9


def test_pythagorean_neutral_runs_returns_one_half():
    p = pythagorean_win_prob(4.5, 4.5)
    assert abs(p - 0.5) < 1e-9


@pytest.fixture
def patched_endpoints(monkeypatch):
    """Stub statsapi.fetch_schedule + Open-Meteo via _cached_get."""
    schedule_bytes = (FIXTURES / "statsapi_schedule.json").read_bytes()
    weather_bytes = (FIXTURES / "openmeteo_sample.json").read_bytes()

    def fake_get(url: str, cache_file: str, **_kwargs):
        if "statsapi.mlb.com" in url:
            return schedule_bytes
        if "api.open-meteo.com" in url:
            return weather_bytes
        return None

    monkeypatch.setattr(scl, "_cached_get", fake_get)
    monkeypatch.setattr(mw, "_cached_get", fake_get)


def test_fetch_events_uses_park_and_weather(patched_endpoints):
    src = MLBWeather()
    events = src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    assert len(events) == 1
    ev = events[0]
    assert ev.sport == "mlb"
    assert "Rockies" in ev.home
    sp = ev.source_probs[0]
    assert sp.source == "mlb-weather"
    assert 0.05 <= sp.home_win_prob <= 0.95
    assert "Coors" in sp.notes
    assert "runs_h=" in sp.notes


def test_filters_other_sports(patched_endpoints):
    src = MLBWeather()
    assert src.fetch_events(date(2026, 5, 30), date(2026, 5, 30), sport="nfl") == []
