"""Unit tests for the CFB market consensus (ESPN scoreboard) connector."""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.sources.cfb_market_consensus import (
    CFBMarketConsensus,
    _extract_moneylines,
    _parse_american,
)


def test_parse_american_handles_int_string_signed():
    assert _parse_american(-150) == -150
    assert _parse_american("+130") == 130
    assert _parse_american("220") == 220
    assert _parse_american(None) is None
    assert _parse_american("not-a-number") is None


def test_extract_moneylines_picks_first_provider():
    odds = [
        {
            "homeTeamOdds": {"moneyLine": -180},
            "awayTeamOdds": {"moneyLine": 160},
        },
        # second provider should be ignored
        {
            "homeTeamOdds": {"moneyLine": -200},
            "awayTeamOdds": {"moneyLine": 170},
        },
    ]
    out = _extract_moneylines(odds)
    assert out == {"home": -180, "away": 160}


def test_extract_moneylines_empty_when_no_data():
    assert _extract_moneylines([]) == {}
    assert _extract_moneylines([{"details": "BAMA -28.5"}]) == {}


def test_connector_returns_empty_outside_cfb():
    assert CFBMarketConsensus().fetch_events(date(2024, 9, 1), date(2024, 9, 7), sport="nfl") == []  # type: ignore[arg-type]


def test_connector_handles_fetch_failure(monkeypatch):
    """When ESPN returns an error, connector yields [] not exception."""
    import httpx

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw):
            raise httpx.ConnectError("network down")

    monkeypatch.setattr("flashcat.sources.cfb_market_consensus.httpx.Client", _FakeClient)
    src = CFBMarketConsensus()
    out = src.fetch_events(date(2024, 9, 1), date(2024, 9, 3), sport="cfb")
    assert out == []


def test_connector_parses_fixture(monkeypatch):
    """ESPN scoreboard fixture → one Event with home/away ML."""
    fixture = {
        "events": [
            {
                "id": "401520415",
                "date": "2024-09-07T19:30:00Z",
                "competitions": [{
                    "competitors": [
                        {"homeAway": "home", "team": {"displayName": "Alabama"}},
                        {"homeAway": "away", "team": {"displayName": "South Florida"}},
                    ],
                    "odds": [{
                        "homeTeamOdds": {"moneyLine": -2500},
                        "awayTeamOdds": {"moneyLine": 1200},
                    }],
                }],
            }
        ]
    }

    class _R:
        status_code = 200
        def json(self): return fixture
        def raise_for_status(self): pass

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url, **kw): return _R()

    monkeypatch.setattr("flashcat.sources.cfb_market_consensus.httpx.Client", _FakeClient)
    src = CFBMarketConsensus()
    out = src.fetch_events(date(2024, 9, 7), date(2024, 9, 7), sport="cfb")
    assert len(out) == 1
    ev = out[0]
    assert ev.sport == "cfb"
    assert ev.home == "Alabama"
    assert ev.away == "South Florida"
    sides = sorted((ln.side, ln.american) for ln in ev.lines)
    assert sides == [("away", 1200), ("home", -2500)]
    assert all(ln.book == "espn-consensus" for ln in ev.lines)
