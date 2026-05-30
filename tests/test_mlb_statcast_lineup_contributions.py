"""Integration tests for the per-batter contribution side-channel.

Confirms the connector (a) populates ``SourceProb.metadata['lineup_contributions']``
with per-batter rows and (b) persists those rows to the
``mlb_lineup_contributions`` table so the explainer can query them later.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from flashcat.sources import mlb_statcast_lineup as scl
from flashcat.sources.mlb_statcast_lineup import (
    MLBStatcastLineup,
    load_lineup_contributions,
)


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def patched_cached_get(monkeypatch):
    schedule_bytes = (FIXTURES / "statsapi_schedule.json").read_bytes()
    savant_bytes = (FIXTURES / "savant_batter_sample.csv").read_bytes()

    def fake_get(url: str, cache_file: str, **_kwargs):
        if "statsapi.mlb.com" in url and "schedule" in url:
            return schedule_bytes
        if "baseballsavant.mlb.com" in url:
            return savant_bytes
        return None

    monkeypatch.setattr(scl, "_cached_get", fake_get)
    return fake_get


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "source_history.db"
    monkeypatch.setattr(scl, "SOURCE_HISTORY_DB_PATH", db_path)
    return db_path


def test_connector_emits_lineup_contributions_metadata(patched_cached_get, isolated_db):
    src = MLBStatcastLineup()
    events = src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    assert len(events) == 1
    ev = events[0]
    sp = ev.source_probs[0]
    assert sp.metadata is not None
    contribs = sp.metadata.get("lineup_contributions")
    assert isinstance(contribs, list)
    # Fixture has 9 home + 9 away batters.
    assert len(contribs) == 18
    home_rows = [r for r in contribs if r["team_side"] == "home"]
    away_rows = [r for r in contribs if r["team_side"] == "away"]
    assert len(home_rows) == 9
    assert len(away_rows) == 9
    # Each row carries the structured fields the explainer expects.
    sample = home_rows[0]
    for key in (
        "batter_id",
        "batter_name",
        "batting_order_position",
        "xwoba_vs_handedness",
        "league_avg_xwoba",
        "pa_weight",
        "contribution_to_team_score",
        "vs_pitcher_hand",
        "team",
    ):
        assert key in sample, f"missing key {key} in contribution row"
    # Names from the fixture should round-trip.
    home_names = {r["batter_name"] for r in home_rows}
    assert "Charlie Blackmon" in home_names
    away_names = {r["batter_name"] for r in away_rows}
    assert "Mookie Betts" in away_names


def test_connector_persists_to_mlb_lineup_contributions_table(
    patched_cached_get, isolated_db
):
    src = MLBStatcastLineup()
    events = src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    assert len(events) == 1
    ev = events[0]
    rows = load_lineup_contributions(ev.event_id, db_path=isolated_db)
    assert len(rows) == 18
    # Round-trip the structural data.
    home_rows = [r for r in rows if r["team_side"] == "home"]
    assert {r["batting_order_position"] for r in home_rows} == set(range(1, 10))
    # Persisted xwoba_observed flag should be set for batters whose
    # Savant CSV returned a value (all 9 in this fixture).
    assert all(r["xwoba_observed"] == 1 for r in home_rows)


def test_persist_is_idempotent_under_repeat_calls(patched_cached_get, isolated_db):
    """Re-running the connector for the same date must not double-insert."""
    src = MLBStatcastLineup()
    src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    rows = load_lineup_contributions(
        "mlb-statcast-lineup:123456", db_path=isolated_db
    )
    assert len(rows) == 18  # NOT 36


def test_connector_survives_db_write_failure(patched_cached_get, monkeypatch, tmp_path):
    """If persistence throws we still emit the event with metadata."""

    def boom(*args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(scl, "persist_lineup_contributions", boom)

    src = MLBStatcastLineup()
    events = src.fetch_events(date(2026, 5, 30), date(2026, 5, 30))
    assert len(events) == 1
    sp = events[0].source_probs[0]
    # Metadata still populated even though the DB write blew up.
    assert sp.metadata is not None
    assert "lineup_contributions" in sp.metadata


def test_per_sport_gate_unaffected_by_metadata():
    """Regression: SourceProb.metadata addition must not break the per-sport\n    LIVE/RESEARCH gate, which iterates source_scoreboard.json and is\n    schema-agnostic about events.\n    """
    from flashcat.build_site import resolve_sport_modes

    sb = {
        "per_sport": {
            "mlb": {
                "blended": {"n_events": 250, "n_bets": 250, "roi": 0.03},
                "sources": {},
            },
            "atp": {
                "blended": {"n_events": 12000, "n_bets": 12000, "roi": -0.07},
                "sources": {},
            },
        }
    }
    modes = resolve_sport_modes(scoreboard=sb)
    assert modes["mlb"]["mode"] == "live"
    assert modes["atp"]["mode"] == "research"
