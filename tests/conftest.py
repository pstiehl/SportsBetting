"""Pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from flashcat.types import BookLine, Event, SourceProb


@pytest.fixture
def now():
    return datetime(2024, 1, 7, 18, 0, tzinfo=timezone.utc)


@pytest.fixture
def basic_event(now):
    return Event(
        event_id="test:1",
        sport="nfl",
        league="NFL",
        home="Chiefs",
        away="Dolphins",
        commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=0.60, captured_at=now),
            SourceProb(source="src-b", home_win_prob=0.70, captured_at=now),
        ],
        lines=[
            BookLine(book="dk", side="home", american=-200, captured_at=now),
            BookLine(book="dk", side="away", american=170, captured_at=now),
            BookLine(book="fd", side="home", american=-190, captured_at=now),
            BookLine(book="fd", side="away", american=165, captured_at=now),
        ],
    )


@pytest.fixture
def coinflip_event(now):
    """Event with blended prob near 0.5 to test tie-breaker."""
    return Event(
        event_id="test:2",
        sport="nfl",
        league="NFL",
        home="Eagles",
        away="Giants",
        commence_time=now,
        source_probs=[
            SourceProb(source="src-a", home_win_prob=0.49, captured_at=now),
            SourceProb(source="src-b", home_win_prob=0.51, captured_at=now),
        ],
        lines=[
            # Home is slight favorite by moneyline (-115 vs +105)
            BookLine(book="dk", side="home", american=-115, captured_at=now),
            BookLine(book="dk", side="away", american=105, captured_at=now),
        ],
    )


@pytest.fixture
def rlm_event(now):
    """Opening vs current lines for an RLM test."""
    open_time = now - timedelta(days=3)
    return Event(
        event_id="test:3",
        sport="nfl",
        league="NFL",
        home="Bills",
        away="Jets",
        commence_time=now,
        source_probs=[SourceProb(source="src-a", home_win_prob=0.60, captured_at=now)],
        lines=[
            # Open: home -150 (60.0%), away +130 (43.5%)
            BookLine(book="dk", side="home", american=-150, captured_at=open_time, is_opening=True),
            BookLine(book="dk", side="away", american=130, captured_at=open_time, is_opening=True),
            # Close: home -120 (54.5%), away +100 (50.0%) — line moved toward away
            BookLine(book="dk", side="home", american=-120, captured_at=now),
            BookLine(book="dk", side="away", american=100, captured_at=now),
            BookLine(book="fd", side="home", american=-115, captured_at=now),
            BookLine(book="fd", side="away", american=-105, captured_at=now),
        ],
    )
