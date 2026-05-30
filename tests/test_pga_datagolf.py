"""Unit tests for the DataGolf PGA connector.

The connector hits a paid-keyed API, so we never call the real endpoint
in tests. We monkey-patch ``_get`` and ``datagolf_api_key`` to return
captured fixture payloads.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from flashcat.sources import pga_datagolf as dg
from flashcat.sources.pga_datagolf import (
    PGADatagolf,
    bradley_terry,
    matchup_prob_from_win_probs,
)


# --- Fixtures --------------------------------------------------------------


PRE_TOURNAMENT_FIXTURE = {
    "event_name": "Charles Schwab Challenge",
    "event_start_date": "2026-05-28",
    "last_updated": "2026-05-26T10:00:00Z",
    "baseline": [
        {"player_name": "Scottie Scheffler", "win": 12.0, "top_5": 35.0,
         "top_10": 50.0, "top_20": 70.0, "make_cut": 92.0},
        {"player_name": "Rory McIlroy", "win": 9.0, "top_5": 30.0,
         "top_10": 48.0, "top_20": 67.0, "make_cut": 90.0},
        {"player_name": "Xander Schauffele", "win": 7.0, "top_5": 25.0,
         "top_10": 40.0, "top_20": 60.0, "make_cut": 88.0},
        {"player_name": "Collin Morikawa", "win": 5.5, "top_5": 22.0,
         "top_10": 38.0, "top_20": 58.0, "make_cut": 86.0},
        # Long-tail player; sub-1% win prob.
        {"player_name": "Ben Griffin", "win": 0.4, "top_5": 3.0,
         "top_10": 6.0, "top_20": 14.0, "make_cut": 55.0},
    ],
}


def _force_key(monkeypatch, value: str | None = "test-key-stub") -> None:
    """Make ``datagolf_api_key()`` return ``value`` regardless of real env."""
    monkeypatch.setattr(dg, "datagolf_api_key", lambda: value)


def _stub_get(monkeypatch, payload):
    """Make ``_get`` return ``payload`` for any call (caching short-circuit)."""
    monkeypatch.setattr(dg, "_get", lambda *a, **k: payload)


# --- Math helpers ----------------------------------------------------------


def test_bradley_terry_symmetric():
    assert bradley_terry(0.1, 0.1) == pytest.approx(0.5, abs=1e-6)


def test_bradley_terry_lopsided():
    # 10x stronger should win ~ 10/11 = ~0.909
    p = bradley_terry(0.9, 0.09)
    assert p > 0.9 and p < 0.95


def test_bradley_terry_zero_safe():
    # Both zero should fall back to 0.5 via smoothing, not divide by zero.
    p = bradley_terry(0.0, 0.0)
    assert p == pytest.approx(0.5, abs=1e-6)


def test_matchup_prob_clamped():
    # If one player has win=0 and the other has a real prob, result clamps
    # but stays in [0.001, 0.999].
    p_win = matchup_prob_from_win_probs(0.0, 0.5)
    assert 0.001 <= p_win <= 0.999
    assert p_win < 0.5


# --- Connector -------------------------------------------------------------


def test_no_key_returns_empty(monkeypatch):
    _force_key(monkeypatch, None)
    src = PGADatagolf()
    events = src.fetch_events(date(2026, 5, 26), date(2026, 5, 30))
    assert events == []


def test_wrong_sport_returns_empty(monkeypatch):
    _force_key(monkeypatch)
    _stub_get(monkeypatch, PRE_TOURNAMENT_FIXTURE)
    src = PGADatagolf()
    out = src.fetch_events(date(2026, 5, 26), date(2026, 5, 30), sport="nfl")
    assert out == []


def test_emits_adjacent_pairs(monkeypatch):
    _force_key(monkeypatch)
    _stub_get(monkeypatch, PRE_TOURNAMENT_FIXTURE)

    src = PGADatagolf()
    events = src.fetch_events(date(2026, 5, 26), date(2026, 5, 30))
    # 5 players → 2 adjacent pairs (the leftover singleton is dropped).
    assert len(events) == 2

    # Each event should carry exactly one DataGolf source prob.
    for e in events:
        assert e.sport == "pga"
        assert len(e.source_probs) == 1
        sp = e.source_probs[0]
        assert sp.source == "datagolf-sg"
        assert 0.0 < sp.home_win_prob < 1.0
        assert "DataGolf" in sp.notes


def test_adjacent_pair_order_scheffler_vs_mcilroy(monkeypatch):
    _force_key(monkeypatch)
    _stub_get(monkeypatch, PRE_TOURNAMENT_FIXTURE)

    events = PGADatagolf().fetch_events(date(2026, 5, 26), date(2026, 5, 30))
    # Top pair: Scheffler (12) vs McIlroy (9) — first event in the list.
    top = events[0]
    names = {top.home, top.away}
    assert "Scottie Scheffler" in names
    assert "Rory McIlroy" in names
    # Scheffler is the favourite (12 vs 9 → BT prob 12/21 ≈ 0.571).
    scheff_prob = (
        top.source_probs[0].home_win_prob
        if top.home == "Scottie Scheffler"
        else 1 - top.source_probs[0].home_win_prob
    )
    assert 0.55 < scheff_prob < 0.62


def test_event_outside_window_skipped(monkeypatch):
    _force_key(monkeypatch)
    _stub_get(monkeypatch, PRE_TOURNAMENT_FIXTURE)

    # Event is on 2026-05-28; ask for a window ending 2026-05-20.
    events = PGADatagolf().fetch_events(date(2026, 5, 14), date(2026, 5, 20))
    assert events == []


def test_baseline_pct_normalization(monkeypatch):
    """DataGolf returns percentages > 1.0 — connector should normalize."""
    _force_key(monkeypatch)
    _stub_get(monkeypatch, PRE_TOURNAMENT_FIXTURE)
    rows = PGADatagolf._extract_baseline_rows(PRE_TOURNAMENT_FIXTURE)
    for r in rows:
        # Every probability field should land in [0, 1].
        for k in ("win", "top_5", "top_10", "top_20", "make_cut"):
            v = r.get(k)
            if v is None:
                continue
            assert 0.0 <= v <= 1.0, f"{k}={v} not in [0,1]"


def test_empty_baseline_returns_empty(monkeypatch):
    _force_key(monkeypatch)
    _stub_get(monkeypatch, {"event_name": "Empty", "baseline": []})
    out = PGADatagolf().fetch_events(date(2026, 1, 1), date(2026, 12, 31))
    assert out == []


def test_load_results_skips_without_key(monkeypatch):
    _force_key(monkeypatch, None)
    out = PGADatagolf().load_results(date(2023, 1, 1), date(2024, 12, 31))
    assert out == []
