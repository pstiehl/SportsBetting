"""Tests for the new Kelly + edge-gate staking module."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from flashcat.model.staking import (
    DEFAULT_EDGE_THRESHOLD,
    TIE_BREAK_BAND_HALF,
    decide_stake,
    devigged_prob,
)
from flashcat.types import BookLine, Event


def _make_event(
    home_american: int = -110, away_american: int = -110, sport: str = "nfl"
) -> Event:
    now = datetime.now(timezone.utc)
    return Event(
        event_id="t:1",
        sport=sport,
        home="A",
        away="B",
        commence_time=now,
        lines=[
            BookLine(book="b", side="home", american=home_american, captured_at=now),
            BookLine(book="b", side="away", american=away_american, captured_at=now),
        ],
    )


def test_devig_split_for_balanced_book():
    ev = _make_event(-110, -110)
    h = devigged_prob(ev, "home")
    a = devigged_prob(ev, "away")
    assert h == pytest.approx(0.5, abs=1e-3)
    assert a == pytest.approx(0.5, abs=1e-3)


def test_no_bet_when_blended_equals_market():
    ev = _make_event(-110, -110)
    d = decide_stake(ev, "home", 0.50)
    assert d.stake == 0
    # Either "within_no_bet_band" or "edge_below_threshold" — both are valid
    # "skip" reasons here; the important invariant is that stake is zero.
    assert d.skipped_reason in {"within_no_bet_band", "edge_below_threshold"}


def test_skip_when_edge_below_threshold():
    ev = _make_event(-110, -110)
    # 2pp edge — under default 3pp threshold
    d = decide_stake(ev, "home", 0.52, no_bet_band=0.0)
    assert d.stake == 0
    assert d.skipped_reason == "edge_below_threshold"


def test_kelly_quarter_sizes_positive_when_edge_exceeds_threshold():
    ev = _make_event(-110, -110)
    d = decide_stake(ev, "home", 0.60, mode="kelly_quarter", bankroll=10_000.0)
    assert d.stake > 0
    assert d.reason == "kelly_quarter"
    # 1/4 Kelly with ~20% full-Kelly fraction (0.6 vs 0.5 market on -110) ~5pp
    # cap should hold the stake below $500 (5% bankroll cap).
    assert d.stake <= 500.0


def test_flat_mode_returns_default_stake():
    ev = _make_event(-110, -110)
    d = decide_stake(ev, "home", 0.60, mode="flat")
    assert d.stake == 100.0
    assert d.reason == "flat"


def test_skip_when_no_market_lines():
    now = datetime.now(timezone.utc)
    ev = Event(
        event_id="t:2",
        sport="atp",
        home="A",
        away="B",
        commence_time=now,
        lines=[],
    )
    d = decide_stake(ev, "home", 0.65)
    assert d.stake == 0
    assert d.skipped_reason in {"no_market_price", "no_market_devig"}


def test_underdog_with_real_edge_gets_a_stake():
    # +200 dog → decimal 3.0. Devigged underdog implied prob ≈ 0.33.
    # Model thinks dog wins 45% → edge ≈ 12pp.
    ev = _make_event(home_american=-250, away_american=+200)
    d = decide_stake(ev, "away", 0.45, mode="kelly_quarter", bankroll=10_000.0)
    assert d.stake > 0
    assert d.edge > DEFAULT_EDGE_THRESHOLD


def test_full_kelly_caps_at_five_percent_of_bankroll():
    # Massive edge — full Kelly would be huge, but we cap at 5%.
    ev = _make_event(home_american=-110, away_american=-110)
    d = decide_stake(ev, "home", 0.95, mode="kelly_full", bankroll=10_000.0)
    assert d.stake <= 500.0  # 5% of $10k
    assert d.stake > 0
