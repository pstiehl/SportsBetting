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
