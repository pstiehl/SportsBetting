"""Unit tests for the CFB ESPN FPI predictor connector."""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.sources.cfb_espn_fpi import (
    CFBESPNFPI,
    _extract_game_projection,
)


def test_extract_game_projection_pulls_value():
    predictor = {
        "homeTeam": {
            "statistics": [
                {"name": "matchupQuality", "value": 92.0},
                {"name": "gameProjection", "value": 78.4},
            ]
        }
    }
    p = _extract_game_projection(predictor)
    assert p == pytest.approx(0.784, abs=1e-6)


def test_extract_game_projection_missing_returns_none():
    assert _extract_game_projection({}) is None
    assert _extract_game_projection({"homeTeam": {"statistics": []}}) is None
    assert _extract_game_projection({
        "homeTeam": {"statistics": [{"name": "other", "value": 50.0}]}
    }) is None


def test_connector_returns_empty_outside_cfb():
    assert CFBESPNFPI().fetch_events(date(2024, 9, 1), date(2024, 9, 7), sport="nba") == []  # type: ignore[arg-type]


def test_connector_handles_scoreboard_failure(monkeypatch):
    import httpx

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            raise httpx.ConnectError("network down")

    monkeypatch.setattr("flashcat.sources.cfb_espn_fpi.httpx.Client", _FakeClient)
    src = CFBESPNFPI()
    out = src.fetch_events(date(2024, 9, 1), date(2024, 9, 3), sport="cfb")
    assert out == []


def test_connector_emits_source_prob_from_fixture(monkeypatch):
    """End-to-end: scoreboard fixture + predictor fixture → one Event w/ FPI source prob."""
    scoreboard = {
        "events": [
            {
                "id": "401520415",
                "date": "2024-09-07T19:30:00Z",
                "competitions": [{
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Alabama"}},
                        {"homeAway": "away", "team": {"displayName": "South Florida"}},
                    ],
                }],
            }
        ]
    }
    predictor = {
        "homeTeam": {
            "statistics": [{"name": "gameProjection", "value": 92.7}]
        }
    }

    class _R:
        def __init__(self, payload): self._p = payload
        status_code = 200
        def json(self): return self._p
        def raise_for_status(self): pass

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            if "scoreboard" in url:
                return _R(scoreboard)
            return _R(predictor)

    monkeypatch.setattr("flashcat.sources.cfb_espn_fpi.httpx.Client", _FakeClient)
    src = CFBESPNFPI()
    out = src.fetch_events(date(2024, 9, 7), date(2024, 9, 7), sport="cfb")
    assert len(out) == 1
    ev = out[0]
    assert ev.sport == "cfb"
    assert ev.home == "Alabama"
    assert len(ev.source_probs) == 1
    sp = ev.source_probs[0]
    assert sp.source == "espn-fpi-cfb"
    assert sp.home_win_prob == pytest.approx(0.927, abs=1e-6)
