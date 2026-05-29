"""Tests for the FanDuel connector.

Golden-file: parses a checked-in sample of the FanDuel content-managed-page
endpoint (MLB) and asserts moneyline extraction + devig math.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashcat.sources.fanduel import FanDuel

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fanduel_mlb():
    with open(FIXTURES / "fanduel_mlb.json") as f:
        return json.load(f)


def test_parse_mlb_extracts_events(fanduel_mlb):
    events = FanDuel.parse_page(fanduel_mlb, sport="mlb")
    assert len(events) >= 10, f"expected >=10 MLB events, got {len(events)}"
    for ev in events:
        assert ev.sport == "mlb"
        assert ev.home and ev.away
        assert len(ev.source_probs) == 1
        sp = ev.source_probs[0]
        assert sp.source == "fanduel"
        assert 0.05 < sp.home_win_prob < 0.95
        assert len(ev.lines) == 2


def test_parse_skips_empty_page():
    assert FanDuel.parse_page({}, sport="mlb") == []
    assert FanDuel.parse_page({"attachments": {}}, sport="mlb") == []
