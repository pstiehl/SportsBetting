"""Tests for backtest math: Brier, ROI, simulate_bet."""

import math
from datetime import datetime, timezone

from flashcat.backtest.grader import brier_score, simulate_bet
from flashcat.types import BookLine, Event, HistoricalResult


def test_brier_perfect_correct():
    assert math.isclose(brier_score(1.0, True), 0.0)
    assert math.isclose(brier_score(0.0, False), 0.0)


def test_brier_perfect_wrong():
    assert math.isclose(brier_score(1.0, False), 1.0)
    assert math.isclose(brier_score(0.0, True), 1.0)


def test_brier_coinflip():
    assert math.isclose(brier_score(0.5, True), 0.25)
    assert math.isclose(brier_score(0.5, False), 0.25)


def test_simulate_bet_win_favorite():
    now = datetime(2024, 1, 7, tzinfo=timezone.utc)
    ev = Event(
        event_id="x",
        sport="nfl",
        home="A",
        away="B",
        commence_time=now,
        lines=[BookLine(book="dk", side="home", american=-200, captured_at=now)],
    )
    res = HistoricalResult(event_id="x", sport="nfl", home="A", away="B",
                           commence_time=now, home_won=True)
    bet = simulate_bet(ev, res, "home", stake=100.0)
    assert bet is not None
    assert bet.won is True
    # -200 wins $50 on $100 stake
    assert math.isclose(bet.profit, 50.0, rel_tol=1e-6)


def test_simulate_bet_lose():
    now = datetime(2024, 1, 7, tzinfo=timezone.utc)
    ev = Event(
        event_id="x",
        sport="nfl",
        home="A",
        away="B",
        commence_time=now,
        lines=[BookLine(book="dk", side="home", american=-200, captured_at=now)],
    )
    res = HistoricalResult(event_id="x", sport="nfl", home="A", away="B",
                           commence_time=now, home_won=False)
    bet = simulate_bet(ev, res, "home", stake=100.0)
    assert bet is not None
    assert bet.won is False
    assert math.isclose(bet.profit, -100.0, rel_tol=1e-6)


def test_simulate_bet_no_line():
    now = datetime(2024, 1, 7, tzinfo=timezone.utc)
    ev = Event(event_id="x", sport="nfl", home="A", away="B",
               commence_time=now, lines=[])
    res = HistoricalResult(event_id="x", sport="nfl", home="A", away="B",
                           commence_time=now, home_won=True)
    bet = simulate_bet(ev, res, "home")
    assert bet is None
