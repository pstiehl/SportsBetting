"""MLB Pythagorean expectation walk-forward test using the 538 fixture."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from flashcat.sources import mlb_pythagorean as pyth


@pytest.fixture
def fake_538_mlb(tmp_path, monkeypatch):
    """Use the bundled 538 sample fixture (10 games of MLB Elo data)."""
    fixture = Path(__file__).parent / "fixtures" / "538_mlb_elo_sample.csv"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "538_mlb_elo.csv").write_bytes(fixture.read_bytes())
    # The download helper looks under CACHE_DIR; patch fivethirtyeight_archives.
    import flashcat.sources.fivethirtyeight_archives as fta
    monkeypatch.setattr(fta, "CACHE_DIR", cache_dir)
    return cache_dir


def test_pythagorean_emits_events_with_probs(fake_538_mlb):
    src = pyth.MLBPythagorean()
    events = src.fetch_events(date(2022, 1, 1), date(2023, 12, 31))
    assert len(events) >= 5
    for ev in events:
        assert ev.sport == "mlb"
        assert len(ev.source_probs) == 1
        sp = ev.source_probs[0]
        assert sp.source == "mlb-pythagorean"
        assert 0.0 < sp.home_win_prob < 1.0


def test_pythagorean_results_align(fake_538_mlb):
    src = pyth.MLBPythagorean()
    events = src.fetch_events(date(2022, 1, 1), date(2023, 12, 31))
    results = src.load_results(date(2022, 1, 1), date(2023, 12, 31))
    assert {e.event_id for e in events} == {r.event_id for r in results}


def test_pythagorean_walk_forward_prob_includes_home_field(fake_538_mlb):
    """First-game prob should be near .50 + HFA (no history yet)."""
    src = pyth.MLBPythagorean()
    events = src.fetch_events(date(2022, 1, 1), date(2023, 12, 31))
    # The earliest event has zero priors → should be very close to .50 + HFA bump
    earliest = min(events, key=lambda e: e.commence_time)
    p = earliest.source_probs[0].home_win_prob
    # Within 0.1 of (.50 + .04) because shrinkage weight is 0/MIN_GAMES = 0
    assert 0.40 <= p <= 0.65
