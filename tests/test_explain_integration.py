"""Integration: build_site renders the 'Why this pick' block for MLB cards."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flashcat.types import BookLine, Event, SourceProb


def _patch_paths(tmp_path, monkeypatch, research: bool = False):
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

    sb_path = tmp_path / "data" / "source_scoreboard.json"
    sb_path.parent.mkdir(parents=True, exist_ok=True)
    if research:
        sb = {
            "window": {"start": "2022-01-01", "end": "2024-12-31", "sport": "multi"},
            "n_events": 1000,
            "sources": {"flashcat-blended": {"n_events": 1000, "roi": -0.04, "brier": 0.22}},
            "per_sport": {
                "mlb": {"n_events": 200, "sources": {}, "blended": {"n_events": 200, "roi": -0.05, "brier": 0.24, "wagered": 1000, "profit": -50, "wins": 95, "losses": 105}},
                "atp": {"n_events": 800, "sources": {}, "blended": {"n_events": 800, "roi": -0.04, "brier": 0.20, "wagered": 8000, "profit": -320, "wins": 380, "losses": 420}},
            },
            "blended_overall": {"n_events": 1000, "wagered": 9000, "profit": -370, "wins": 475, "losses": 525, "roi": -0.04},
        }
    else:
        sb = {
            "window": {"start": "2022-01-01", "end": "2024-12-31", "sport": "multi"},
            "n_events": 500,
            "sources": {"flashcat-blended": {"n_events": 500, "roi": 0.05, "brier": 0.22}},
            "per_sport": {
                "mlb": {"n_events": 200, "sources": {}, "blended": {"n_events": 200, "roi": 0.04, "brier": 0.24, "wagered": 1000, "profit": 40, "wins": 105, "losses": 95}},
            },
            "blended_overall": {"n_events": 200, "wagered": 1000, "profit": 40, "wins": 105, "losses": 95, "roi": 0.04},
        }
    sb_path.write_text(json.dumps(sb))
    monkeypatch.setattr(cfg, "SOURCE_SCOREBOARD_PATH", sb_path)
    monkeypatch.setattr(bs, "SOURCE_SCOREBOARD_PATH", sb_path)
    return tmp_docs, tmp_events


def _make_mlb_event():
    now = datetime(2026, 5, 30, 23, 5, tzinfo=timezone.utc)
    return Event(
        event_id="mlb:test:1",
        sport="mlb",
        league="MLB",
        home="Colorado Rockies",
        away="Los Angeles Dodgers",
        commence_time=now,
        source_probs=[
            SourceProb(source="mlb-statcast-lineup", home_win_prob=0.58, captured_at=now,
                       notes="home_off=0.1100 away_off=0.0900 diff=+0.0200"),
            SourceProb(source="mlb-weather", home_win_prob=0.54, captured_at=now,
                       notes="park=Coors Field dome=False runs_h=5.20 runs_a=5.20 temp=82F wind=15mph dir=180deg"),
            SourceProb(source="mlb-pythagorean", home_win_prob=0.52, captured_at=now,
                       notes="pyth-baseline"),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=now),
            BookLine(book="dk", side="away", american=-110, captured_at=now),
        ],
        blended_home_prob=0.56,
        pick="home",
        pick_prob=0.56,
    )


def test_mlb_event_page_renders_why_this_pick(tmp_path, monkeypatch):
    from flashcat import build_site as bs
    tmp_docs, tmp_events = _patch_paths(tmp_path, monkeypatch, research=True)

    ev = _make_mlb_event()
    bs.build([ev])

    event_files = list(tmp_events.glob("*.html"))
    assert len(event_files) == 1
    html = event_files[0].read_text()
    assert "Why this pick" in html
    assert "Statcast lineup edge" in html
    assert "Weather" in html


def test_mlb_event_card_on_index_includes_rationale(tmp_path, monkeypatch):
    from flashcat import build_site as bs
    tmp_docs, _ = _patch_paths(tmp_path, monkeypatch, research=True)
    ev = _make_mlb_event()
    bs.build([ev])
    index = (tmp_docs / "index.html").read_text()
    assert "Why this pick" in index
    assert "Statcast lineup edge" in index
