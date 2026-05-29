"""Golden-file tests for the 538 archive connectors (MLB / NFL / NBA modern)."""

from datetime import date
from pathlib import Path

import pytest

from flashcat.sources.fivethirtyeight_archives import (
    FiveThirtyEightMLBElo,
    FiveThirtyEightNBAModern,
    FiveThirtyEightNFLElo,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _stage_fixture(fixture_name: str, cache_file: str, tmp_cache: Path) -> None:
    """Copy a CSV fixture into the cache dir so the connector reads it offline."""
    tmp_cache.mkdir(parents=True, exist_ok=True)
    src = FIXTURES / fixture_name
    dst = tmp_cache / cache_file
    dst.write_bytes(src.read_bytes())


@pytest.fixture
def patched_cache(tmp_path, monkeypatch):
    from flashcat import config as cfg
    from flashcat.sources import fivethirtyeight_archives as fa

    tmp_cache = tmp_path / "cache"
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_cache)
    monkeypatch.setattr(fa, "CACHE_DIR", tmp_cache)
    return tmp_cache


def test_mlb_elo_parses_fixture(patched_cache):
    _stage_fixture("538_mlb_elo_sample.csv", "538_mlb_elo.csv", patched_cache)
    c = FiveThirtyEightMLBElo()
    events, results = c._load_range(date(2022, 4, 1), date(2022, 4, 30))
    assert len(events) >= 5
    assert len(results) == len(events)
    sample = events[0]
    src_names = {p.source for p in sample.source_probs}
    # Each event should carry both the team-Elo and pitcher-adjusted rating
    # source probabilities.
    assert "fivethirtyeight-mlb-elo" in src_names
    assert "fivethirtyeight-mlb-rating" in src_names
    # Probabilities are bounded.
    for p in sample.source_probs:
        assert 0.0 < p.home_win_prob < 1.0
    # Event id is stable and scoped to the sport.
    assert sample.event_id.startswith("538mlb:")
    # All events are MLB.
    assert all(e.sport == "mlb" for e in events)


def test_mlb_elo_sport_filter(patched_cache):
    _stage_fixture("538_mlb_elo_sample.csv", "538_mlb_elo.csv", patched_cache)
    c = FiveThirtyEightMLBElo()
    assert c.fetch_events(date(2022, 4, 1), date(2022, 4, 30), sport="nfl") == []


def test_nfl_elo_parses_fixture(patched_cache):
    _stage_fixture("538_nfl_elo_sample.csv", "538_nfl_elo.csv", patched_cache)
    c = FiveThirtyEightNFLElo()
    events, results = c._load_range(date(2022, 9, 1), date(2022, 12, 31))
    assert events, "expected at least one event in NFL fixture window"
    assert len(results) == len(events)
    src_names = {p.source for ev in events for p in ev.source_probs}
    assert "fivethirtyeight-nfl-elo" in src_names
    assert "fivethirtyeight-nfl-qbelo" in src_names
    assert all(e.sport == "nfl" for e in events)


def test_nba_modern_parses_fixture(patched_cache):
    _stage_fixture("538_nba_elo_sample.csv", "538_nba_elo.csv", patched_cache)
    c = FiveThirtyEightNBAModern()
    events, results = c._load_range(date(2022, 1, 1), date(2022, 1, 31))
    assert events, "expected at least one event in NBA fixture window"
    assert len(results) == len(events)
    src_names = {p.source for ev in events for p in ev.source_probs}
    assert "fivethirtyeight-nba-elo-modern" in src_names
    # 2022 fixture window includes RAPTOR + CARM data.
    assert "fivethirtyeight-nba-raptor" in src_names
    assert all(e.sport == "nba" for e in events)


def test_archive_returns_empty_when_cache_missing(tmp_path, monkeypatch):
    """No network, no cache → empty list (don't crash the backtest)."""
    from flashcat import config as cfg
    from flashcat.sources import fivethirtyeight_archives as fa

    empty_cache = tmp_path / "empty"
    monkeypatch.setattr(cfg, "CACHE_DIR", empty_cache)
    monkeypatch.setattr(fa, "CACHE_DIR", empty_cache)
    # Force the network call to fail by pointing at an unreachable URL.
    monkeypatch.setattr(fa, "MLB_URL", "http://127.0.0.1:1/no")
    c = FiveThirtyEightMLBElo(timeout=0.5)
    events = c.fetch_events(date(2022, 1, 1), date(2022, 12, 31), sport="mlb")
    assert events == []
