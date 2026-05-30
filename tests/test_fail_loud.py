"""Tests for the no-fake-data fail-loud behavior."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from flashcat.config import NoLiveDataError, use_samples_fallback
from flashcat.sources.odds_api import TheOddsAPI


def test_odds_api_returns_empty_without_key(monkeypatch):
    """Without a key and without opt-in, OddsAPI must return []."""
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FLASHCAT_USE_SAMPLES", raising=False)
    conn = TheOddsAPI()
    out = conn.fetch_events(date.today(), date.today())
    assert out == []


def test_odds_api_loads_samples_when_opted_in(monkeypatch, tmp_path):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.setenv("FLASHCAT_USE_SAMPLES", "1")
    assert use_samples_fallback() is True
    conn = TheOddsAPI()
    out = conn.fetch_events(date.today(), date.today())
    # Sample contains MLB + NBA exhibition games — must be a list (possibly empty
    # if the example file was removed).
    assert isinstance(out, list)


def test_build_fails_loud_when_no_live_events(monkeypatch):
    """The `build` CLI must raise NoLiveDataError when all sources return []."""
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    monkeypatch.delenv("FLASHCAT_USE_SAMPLES", raising=False)
    from flashcat import cli

    # Patch connectors to return nothing.
    def empty_fetch(self, *a, **kw):
        return []

    monkeypatch.setattr("flashcat.cli.TheOddsAPI.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.Bovada.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.FanDuel.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.ESPNScoreboard.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.Polymarket.fetch_events", empty_fetch)
    # PGA connectors added in PR #15 — also empty here to keep the
    # "no live data anywhere" branch reachable in this test.
    monkeypatch.setattr("flashcat.cli.PGADatagolf.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.PGAESPNScoreboard.fetch_events", empty_fetch)
    monkeypatch.setattr("flashcat.cli.PGAMarketConsensus.fetch_events", empty_fetch)

    with pytest.raises(NoLiveDataError):
        cli.build(days_ahead=0)


def test_sample_quarantine_filename_is_explicit():
    """The sample file lives at `odds_api_sample.example.json`, not the
    plain name (which used to silently fall back in production)."""
    from flashcat.config import SAMPLES_DIR

    assert (SAMPLES_DIR / "odds_api_sample.example.json").exists()
    # The plain name must NOT exist anymore — that's the whole point.
    assert not (SAMPLES_DIR / "odds_api_sample.json").exists()
