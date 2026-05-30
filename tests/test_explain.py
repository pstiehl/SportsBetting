"""Tests for the per-pick rationale generator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from flashcat.explain import explain_event
from flashcat.types import BookLine, Event, SourceProb


def _ts():
    return datetime(2026, 5, 30, 18, 0, tzinfo=timezone.utc)


def test_statcast_signal_renders_runs_per_game():
    ev = Event(
        event_id="t:1",
        sport="mlb",
        league="MLB",
        home="Rockies",
        away="Dodgers",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-statcast-lineup",
                home_win_prob=0.58,
                captured_at=_ts(),
                notes="home_off=0.1100 away_off=0.0900 diff=+0.0200",
            ),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
    )
    out = explain_event(ev)
    assert any("Statcast lineup edge" in s for s in out)
    assert any("ROCKIES" in s for s in out)


def test_weather_outdoor_includes_run_environment_pct():
    ev = Event(
        event_id="t:2",
        sport="mlb",
        league="MLB",
        home="Rockies",
        away="Dodgers",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-weather",
                home_win_prob=0.55,
                captured_at=_ts(),
                notes="park=Coors Field dome=False runs_h=5.20 runs_a=5.20 temp=82F wind=12mph dir=180deg",
            ),
        ],
    )
    out = explain_event(ev)
    assert any("Weather" in s and "run environment" in s for s in out)


def test_weather_dome_emits_no_weather_explanation():
    ev = Event(
        event_id="t:3",
        sport="mlb",
        league="MLB",
        home="Rays",
        away="Yankees",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-weather",
                home_win_prob=0.5,
                captured_at=_ts(),
                notes="park=Tropicana Field dome=True runs_h=4.5 runs_a=4.5",
            ),
        ],
    )
    out = explain_event(ev)
    assert not any("Weather" in s for s in out)


def test_market_consensus_explanation_when_lines_present():
    ev = Event(
        event_id="t:4",
        sport="mlb",
        league="MLB",
        home="Rockies",
        away="Dodgers",
        commence_time=_ts(),
        source_probs=[
            SourceProb(
                source="mlb-statcast-lineup",
                home_win_prob=0.58,
                captured_at=_ts(),
                notes="home_off=0.11 away_off=0.09 diff=+0.02",
            ),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=_ts()),
            BookLine(book="dk", side="away", american=-110, captured_at=_ts()),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
    )
    out = explain_event(ev)
    assert any("Market consensus" in s for s in out)


def test_caps_at_top_n():
    ev = Event(
        event_id="t:5",
        sport="mlb",
        league="MLB",
        home="Rockies",
        away="Dodgers",
        commence_time=_ts(),
        source_probs=[
            SourceProb(source="mlb-statcast-lineup", home_win_prob=0.6, captured_at=_ts(),
                       notes="home_off=0.10 away_off=0.08 diff=+0.020"),
            SourceProb(source="mlb-weather", home_win_prob=0.55, captured_at=_ts(),
                       notes="park=Wrigley Field dome=False runs_h=5 runs_a=5 temp=72F wind=12mph dir=180deg"),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-110, captured_at=_ts()),
            BookLine(book="dk", side="away", american=-110, captured_at=_ts()),
        ],
        blended_home_prob=0.58,
        pick="home",
        pick_prob=0.58,
        signals=["chalk-overpriced"],
    )
    out = explain_event(ev, top_n=2)
    assert len(out) == 2
