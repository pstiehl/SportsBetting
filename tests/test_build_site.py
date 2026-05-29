"""End-to-end smoke test: site build runs without errors on sample inputs."""

import json
from datetime import datetime, timezone
from pathlib import Path

from flashcat.build_site import build
from flashcat.config import DOCS_DIR, SOURCE_SCOREBOARD_PATH
from flashcat.types import BookLine, Event, SourceProb


def test_site_build_smoke(tmp_path, monkeypatch):
    # Use a temp docs dir so tests don't blow over the real site
    from flashcat import build_site as bs
    from flashcat import config as cfg

    tmp_docs = tmp_path / "docs"
    tmp_assets = tmp_docs / "assets"
    tmp_events = tmp_docs / "event"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_events)
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_events)

    # Sample event
    now = datetime(2024, 1, 7, 18, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="test:1",
        sport="nfl",
        home="Chiefs",
        away="Dolphins",
        commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=0.65, captured_at=now),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-200, captured_at=now),
            BookLine(book="dk", side="away", american=170, captured_at=now),
        ],
        blended_home_prob=0.65,
        pick="home",
        pick_prob=0.65,
    )

    bs.build([ev])
    assert (tmp_docs / "index.html").exists()
    assert (tmp_docs / "source-scoreboard.html").exists()
    assert (tmp_docs / "methodology.html").exists()
    assert (tmp_docs / "backtest.html").exists()
    assert (tmp_assets / "flashcat-logo.svg").exists()
    # Event page exists
    event_pages = list(tmp_events.glob("*.html"))
    assert len(event_pages) == 1

    index_html = (tmp_docs / "index.html").read_text()
    # New stake-aware card copy replaces the old "$100 flat" hardcode.
    assert "$100 flat \u00b7" not in index_html
    assert "Recommended stake" in index_html
    assert "Model edge" in index_html
    # Local-time conversion: ISO datetime attribute is emitted for the visitor's
    # browser to convert client-side.
    assert 'data-local="show"' in index_html
    assert "2024-01-07T18:00:00Z" in index_html
    # Recommended plays section is always rendered (with empty-state fallback).
    assert "Recommended Plays Today" in index_html

    # Layout: inline local-time conversion script lives on every page.
    assert "toLocaleString" in index_html


def test_recommended_plays_panel_shows_edge_plays(tmp_path, monkeypatch):
    """Index renders a ranked Recommended Plays table when edge clears threshold."""
    from flashcat import build_site as bs
    from flashcat import config as cfg

    tmp_docs = tmp_path / "docs"
    tmp_assets = tmp_docs / "assets"
    tmp_events = tmp_docs / "event"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_events)
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_events)

    now = datetime(2024, 1, 7, 18, 0, tzinfo=timezone.utc)
    # Pick home, blended 0.60 vs devigged-market ~0.50 (-110/-110) → edge ~10pp
    edge_event = Event(
        event_id="edge:1",
        sport="nfl",
        home="Sharks",
        away="Bears",
        commence_time=now,
        source_probs=[SourceProb(source="src-a", home_win_prob=0.60, captured_at=now)],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=now),
            BookLine(book="dk", side="away", american=-110, captured_at=now),
        ],
        blended_home_prob=0.60,
        pick="home",
        pick_prob=0.60,
    )
    bs.build([edge_event])
    html = (tmp_docs / "index.html").read_text()
    assert "Recommended Plays Today" in html
    # No sit-out copy when an edge play exists.
    assert "sitting out" not in html
    # The table includes the pick row.
    assert "SHARKS" in html


def test_recommended_plays_sit_out_fallback(tmp_path, monkeypatch):
    """When no event clears the edge threshold, panel shows the sit-out callout."""
    from flashcat import build_site as bs
    from flashcat import config as cfg

    tmp_docs = tmp_path / "docs"
    tmp_assets = tmp_docs / "assets"
    tmp_events = tmp_docs / "event"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_events)
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_events)

    now = datetime(2024, 1, 7, 18, 0, tzinfo=timezone.utc)
    # Blended ~0.51 vs market ~0.50 → edge well below threshold.
    no_edge_event = Event(
        event_id="flat:1",
        sport="nfl",
        home="A",
        away="B",
        commence_time=now,
        source_probs=[SourceProb(source="src-a", home_win_prob=0.51, captured_at=now)],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=now),
            BookLine(book="dk", side="away", american=-110, captured_at=now),
        ],
        blended_home_prob=0.51,
        pick="home",
        pick_prob=0.51,
    )
    bs.build([no_edge_event])
    html = (tmp_docs / "index.html").read_text()
    assert "sitting out" in html
    # Card itself should show a no-bet line for this event (the exact phrasing
    # depends on which gate caught it: within_no_bet_band or edge_below_threshold).
    assert ("NO BET" in html) or ("no bet" in html) or ("Coin flip" in html)
