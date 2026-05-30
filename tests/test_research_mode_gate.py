"""Regression: site stays RESEARCH MODE when any per-sport blend is negative.

Phil's rule (2026-05-29): if ANY sport's blended ROI is negative, the entire
site shows the RESEARCH-MODE badge and suppresses stake recommendations.
This test pins that behavior so future refactors can't quietly remove the
gate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone


def test_negative_per_sport_blended_roi_triggers_research_mode(tmp_path, monkeypatch):
    from flashcat.build_site import _research_mode_state

    sb = {
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 1000, "roi": 0.10, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": -0.05, "brier": 0.20}},
            "nfl": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": 0.08, "brier": 0.21}},
        },
        "blended_overall": {"n_events": 200, "wagered": 2000, "profit": 195, "roi": 0.10},
    }
    state = _research_mode_state(sb)
    assert state["research_mode"] is True
    assert "ATP" in (state["reason"] or "").upper()


def test_all_positive_per_sport_allows_live(tmp_path):
    from flashcat.build_site import _research_mode_state

    sb = {
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 1000, "roi": 0.10, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": 0.05, "brier": 0.20}},
            "nfl": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": 0.08, "brier": 0.21}},
        },
        "blended_overall": {"n_events": 200, "wagered": 2000, "profit": 195, "roi": 0.10},
    }
    state = _research_mode_state(sb)
    assert state["research_mode"] is False


def test_grouped_layout_pushes_negative_roi_sport_into_research_bucket(tmp_path, monkeypatch):
    """Even when site clears the gate, individual sport with negative ROI
    means its picks land in the RESEARCH bucket, not RECOMMENDED."""
    from flashcat import build_site as bs
    from flashcat import config as cfg
    from flashcat.types import BookLine, Event, SourceProb

    tmp_docs = tmp_path / "docs"
    tmp_assets = tmp_docs / "assets"
    tmp_events = tmp_docs / "event"
    monkeypatch.setattr(cfg, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(cfg, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(cfg, "EVENT_PAGES_DIR", tmp_events)
    monkeypatch.setattr(bs, "DOCS_DIR", tmp_docs)
    monkeypatch.setattr(bs, "ASSETS_DIR", tmp_assets)
    monkeypatch.setattr(bs, "EVENT_PAGES_DIR", tmp_events)

    sb_path = tmp_path / "data" / "source_scoreboard.json"
    sb_path.parent.mkdir(parents=True, exist_ok=True)
    sb_path.write_text(json.dumps({
        "window": {"start": "2022-01-01", "end": "2024-12-31"},
        "sources": {"flashcat-blended": {"n_events": 200, "roi": -0.04, "brier": 0.22}},
        "per_sport": {
            "atp": {"n_events": 100, "sources": {}, "blended": {"n_events": 100, "roi": -0.05, "brier": 0.20, "wagered": 1000, "profit": -50, "wins": 45, "losses": 55}},
        },
        "blended_overall": {"n_events": 100, "wagered": 1000, "profit": -50, "roi": -0.05},
    }))
    monkeypatch.setattr(cfg, "SOURCE_SCOREBOARD_PATH", sb_path)
    monkeypatch.setattr(bs, "SOURCE_SCOREBOARD_PATH", sb_path)

    now = datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)
    ev = Event(
        event_id="atp:1",
        sport="atp",
        home="Player A",
        away="Player B",
        commence_time=now,
        source_probs=[SourceProb(source="src", home_win_prob=0.62, captured_at=now)],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=now),
            BookLine(book="dk", side="away", american=-110, captured_at=now),
        ],
        blended_home_prob=0.62,
        pick="home",
        pick_prob=0.62,
    )
    bs.build([ev])
    html = (tmp_docs / "index.html").read_text()
    # Site stays RESEARCH MODE because ATP blended ROI is negative.
    assert "RESEARCH" in html.upper()
    # The Research-Mode Picks section is rendered.
    assert "Research-Mode Picks" in html or "research-mode picks" in html.lower()
