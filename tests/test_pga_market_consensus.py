"""Unit tests for the PGA market consensus connector."""

from __future__ import annotations

from datetime import date

import pytest

from flashcat.sources import pga_market_consensus as pmc
from flashcat.sources.pga_market_consensus import PGAMarketConsensus


SPORTS_LIST_FIXTURE = [
    {"key": "golf_masters_tournament_winner", "active": False},
    {"key": "golf_pga_championship_winner", "active": True,
     "title": "PGA Championship Winner"},
    {"key": "golf_us_open_winner", "active": True,
     "title": "US Open Winner"},
    {"key": "americanfootball_nfl", "active": True},
]


OUTRIGHT_FIXTURE = [
    {
        "id": "abc123",
        "sport_key": "golf_pga_championship_winner",
        "sport_title": "PGA Championship Winner",
        "commence_time": "2026-05-14T12:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "outrights",
                        "outcomes": [
                            {"name": "Scottie Scheffler", "price": 700},
                            {"name": "Rory McIlroy", "price": 900},
                            {"name": "Xander Schauffele", "price": 1200},
                        ],
                    }
                ],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "outrights",
                        "outcomes": [
                            {"name": "Scottie Scheffler", "price": 750},
                            {"name": "Rory McIlroy", "price": 850},
                        ],
                    }
                ],
            },
        ],
    }
]


def _force_key(monkeypatch, value: str | None = "test-key"):
    monkeypatch.setattr(pmc, "the_odds_api_key", lambda: value)


def test_no_key_returns_empty(monkeypatch):
    _force_key(monkeypatch, None)
    out = PGAMarketConsensus().fetch_events(
        date(2026, 5, 1), date(2026, 5, 31)
    )
    assert out == []


def test_wrong_sport_returns_empty(monkeypatch):
    _force_key(monkeypatch)
    out = PGAMarketConsensus().fetch_events(
        date(2026, 5, 1), date(2026, 5, 31), sport="nba"
    )
    assert out == []


def test_no_active_pga_keys_returns_empty(monkeypatch):
    _force_key(monkeypatch)
    # All golf keys inactive.
    monkeypatch.setattr(
        PGAMarketConsensus, "_active_pga_keys",
        lambda self, api_key: []
    )
    out = PGAMarketConsensus().fetch_events(
        date(2026, 5, 1), date(2026, 5, 31)
    )
    assert out == []


def test_active_keys_filtered(monkeypatch):
    _force_key(monkeypatch)
    monkeypatch.setattr(
        PGAMarketConsensus, "_fetch_outright",
        lambda self, key, sk: []
    )
    # Patch _active_pga_keys's inner http call.
    import httpx as _httpx

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return SPORTS_LIST_FIXTURE

    class _FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, *a, **k): return _FakeResp()

    monkeypatch.setattr(_httpx, "Client", _FakeClient)

    active = PGAMarketConsensus()._active_pga_keys("k")
    assert "golf_pga_championship_winner" in active
    assert "golf_us_open_winner" in active
    assert "golf_masters_tournament_winner" not in active  # inactive
    assert "americanfootball_nfl" not in active  # not in PGA list


def test_parse_outright_emits_one_event_per_player_per_book():
    events = PGAMarketConsensus._parse_outright(
        OUTRIGHT_FIXTURE, "golf_pga_championship_winner"
    )
    # 3 players × draftkings + 2 players × fanduel = 5 events.
    assert len(events) == 5
    for e in events:
        assert e.sport == "pga"
        assert e.away == "Field"
        assert len(e.lines) == 1
        ln = e.lines[0]
        assert ln.side == "home"
        assert isinstance(ln.american, int)


def test_parse_outright_keeps_player_name_in_home():
    events = PGAMarketConsensus._parse_outright(
        OUTRIGHT_FIXTURE, "golf_pga_championship_winner"
    )
    names = {e.home for e in events}
    assert "Scottie Scheffler" in names
    assert "Rory McIlroy" in names
    # Scheffler appears in both books.
    scheff = [e for e in events if e.home == "Scottie Scheffler"]
    assert len(scheff) == 2
    books = {e.lines[0].book for e in scheff}
    assert books == {"draftkings", "fanduel"}


def test_parse_outright_skips_non_int_price():
    payload = [
        {
            "id": "x",
            "sport_key": "golf_us_open_winner",
            "sport_title": "US Open Winner",
            "commence_time": "2026-06-18T12:00:00Z",
            "bookmakers": [
                {
                    "key": "bovada",
                    "markets": [
                        {
                            "key": "outrights",
                            "outcomes": [
                                {"name": "Player Bad Price", "price": "n/a"},
                                {"name": "Player Good Price", "price": 500},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    events = PGAMarketConsensus._parse_outright(payload, "golf_us_open_winner")
    assert len(events) == 1
    assert events[0].home == "Player Good Price"
