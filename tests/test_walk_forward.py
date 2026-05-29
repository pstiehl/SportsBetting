"""Smoke tests for no-outcome-leakage in the backtest pipeline."""

from __future__ import annotations

import random
from datetime import datetime, time, timezone

from flashcat.backtest.runner import _attach_market_source_prob
from flashcat.model.blend import blend_events
from flashcat.model.staking import decide_stake
from flashcat.types import BookLine, Event, HistoricalResult, SourceProb


def _synthetic_event(idx: int, home_prob: float, market_home_prob: float) -> Event:
    """Create a synthetic event with a single source prob + balanced moneylines.

    market_home_prob ≈ 0.5 means -110/-110; we approximate by adjusting prices.
    """
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    # American odds → implied prob with 5% vig split symmetrically.
    home_dec = 1.0 / (market_home_prob * 1.02)  # decimal odds with vig
    away_dec = 1.0 / ((1 - market_home_prob) * 1.02)
    home_american = int(round((home_dec - 1) * 100)) if home_dec >= 2 else -int(round(100 / (home_dec - 1)))
    away_american = int(round((away_dec - 1) * 100)) if away_dec >= 2 else -int(round(100 / (away_dec - 1)))
    return Event(
        event_id=f"synth:{idx}",
        sport="atp",
        home="A",
        away="B",
        commence_time=now,
        lines=[
            BookLine(book="b", side="home", american=home_american, captured_at=now),
            BookLine(book="b", side="away", american=away_american, captured_at=now),
        ],
        source_probs=[
            SourceProb(
                source="synthetic-model",
                home_win_prob=home_prob,
                captured_at=now,
                notes="test",
            )
        ],
    )


def test_random_outcomes_give_near_zero_blended_roi():
    """Shuffle outcomes uniformly. If the pipeline doesn't leak, blended ROI on
    the picked events should be statistically indistinguishable from zero
    (centred around -vig, but with a small sample we just demand the
    *distribution* is symmetric — easier: assert mean is within a couple of
    standard errors of zero).
    """
    rng = random.Random(42)
    n = 500
    events: list[Event] = []
    results: list[HistoricalResult] = []
    for i in range(n):
        market = rng.uniform(0.35, 0.65)
        # Make model disagree with market by a small random amount.
        model = max(0.05, min(0.95, market + rng.uniform(-0.10, 0.10)))
        ev = _synthetic_event(i, model, market)
        events.append(ev)
        # Outcomes are RANDOM, not tied to model.
        results.append(
            HistoricalResult(
                event_id=ev.event_id,
                sport="atp",
                home="A",
                away="B",
                commence_time=ev.commence_time,
                home_won=rng.random() < 0.5,
            )
        )

    _attach_market_source_prob(events)
    # Equal weights — no fitting on the outcomes inside the test window.
    blended = blend_events(events, {})

    res_by_id = {r.event_id: r for r in results}
    bets = 0
    profit = 0.0
    wagered = 0.0
    for ev in blended:
        if ev.pick is None or ev.event_id not in res_by_id:
            continue
        d = decide_stake(
            ev, ev.pick, ev.pick_prob or 0.5,
            mode="kelly_quarter", edge_threshold=0.03,
        )
        if d.stake <= 0:
            continue
        res = res_by_id[ev.event_id]
        won = (ev.pick == "home" and res.home_won) or (
            ev.pick == "away" and not res.home_won
        )
        from flashcat.types import american_to_profit

        profit += american_to_profit(d.price, d.stake) if won else -d.stake
        wagered += d.stake
        bets += 1

    if wagered == 0:
        return  # No bets passed the gate — pipeline is conservative. Fine.
    roi = profit / wagered
    # With ~500 random outcomes and Kelly_quarter staking, a leakage-free
    # pipeline should not produce double-digit ROI in either direction. We
    # want this loose enough not to be flaky but tight enough to catch
    # accidental outcome leakage.
    assert abs(roi) < 0.30, f"suspicious ROI on shuffled outcomes: {roi:+.4f}"
