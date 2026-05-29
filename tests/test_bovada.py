"""Tests for the Bovada connector.

Golden-file: parses checked-in samples of the live Bovada coupon endpoint
and asserts we extract a sensible number of events with devigged moneyline
probabilities.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from flashcat.sources.bovada import Bovada, devig_two_way

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def bovada_mlb():
    with open(FIXTURES / "bovada_mlb.json") as f:
        return json.load(f)


@pytest.fixture
def bovada_french_open():
    with open(FIXTURES / "bovada_french_open.json") as f:
        return json.load(f)


def test_devig_marlins_mets():
    """Marlins -104 (away) vs Mets -112 (home) devigs to ~50.9% home."""
    p = devig_two_way(home_american=-112, away_american=-104)
    assert math.isclose(p, 0.5089, abs_tol=0.001)


def test_devig_dbacks_mariners():
    """Diamondbacks +122 (away) vs Mariners -144 (home) → ~56.7% home."""
    p = devig_two_way(home_american=-144, away_american=122)
    assert math.isclose(p, 0.5671, abs_tol=0.002)


def test_devig_bounded():
    assert 0.0 < devig_two_way(-1000, 700) < 1.0
    assert 0.0 < devig_two_way(700, -1000) < 1.0


def test_parse_mlb_extracts_events(bovada_mlb):
    events = Bovada.parse_coupon(bovada_mlb, fallback_sport="mlb")
    assert len(events) >= 10, f"expected >=10 MLB events, got {len(events)}"
    for ev in events:
        assert ev.sport == "mlb"
        assert ev.home and ev.away
        assert len(ev.source_probs) == 1
        sp = ev.source_probs[0]
        assert sp.source == "bovada"
        assert 0.05 < sp.home_win_prob < 0.95
        # Both BookLines present
        assert len(ev.lines) == 2
        sides = sorted(l.side for l in ev.lines)
        assert sides == ["away", "home"]


def test_parse_french_open_extracts_singles(bovada_french_open):
    events = Bovada.parse_coupon(bovada_french_open)
    # Doubles + mixed should be skipped — only atp and wta singles remain.
    sports = {e.sport for e in events}
    assert sports.issubset({"atp", "wta"})
    assert "atp" in sports
    assert "wta" in sports
    # At least a few of each.
    atp_count = sum(1 for e in events if e.sport == "atp")
    wta_count = sum(1 for e in events if e.sport == "wta")
    assert atp_count >= 5
    assert wta_count >= 5
    for ev in events:
        assert 0.001 < ev.source_probs[0].home_win_prob < 0.999


def test_event_ids_are_unique(bovada_mlb, bovada_french_open):
    events = Bovada.parse_coupon(bovada_mlb, fallback_sport="mlb") + Bovada.parse_coupon(
        bovada_french_open
    )
    ids = [e.event_id for e in events]
    assert len(ids) == len(set(ids)), "duplicate event ids"
