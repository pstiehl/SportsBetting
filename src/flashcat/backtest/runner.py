"""Backtest runner — pulls historical data, blends, simulates, writes scoreboard."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

from ..config import (
    DOCS_DIR,
    FLAT_STAKE,
    SOURCE_SCOREBOARD_PATH,
)
from ..model.blend import blend_events, load_weights
from ..signals.favlong import detect as detect_favlong
from ..signals.sharp import detect as detect_sharp
from ..sources.nflverse import NFLverseHistorical
from ..types import (
    Event,
    HistoricalResult,
    SourceProb,
    Side,
    american_to_prob,
    devig_two_way,
)
from .grader import brier_score, build_scoreboard, simulate_bet

log = logging.getLogger(__name__)


def _attach_market_source_prob(events: list[Event]) -> None:
    """Synthesize a `market-close` SourceProb on each event from devigged closing lines.

    This lets the blender include the market line as one of its sources.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for ev in events:
        if not ev.lines:
            continue
        home_prices = [
            american_to_prob(ln.american)
            for ln in ev.lines
            if ln.side == "home" and not ln.is_opening
        ]
        away_prices = [
            american_to_prob(ln.american)
            for ln in ev.lines
            if ln.side == "away" and not ln.is_opening
        ]
        if not home_prices or not away_prices:
            continue
        h = sum(home_prices) / len(home_prices)
        a = sum(away_prices) / len(away_prices)
        h_devig, _ = devig_two_way(h, a)
        if not any(p.source == "market-close" for p in ev.source_probs):
            ev.source_probs.append(
                SourceProb(
                    source="market-close",
                    home_win_prob=h_devig,
                    captured_at=now,
                    notes="devigged consensus close moneyline",
                )
            )


def run_backtest(
    start: date,
    end: date,
    sport: str = "nfl",
    output_path: Path | None = None,
) -> dict:
    """Run the full backtest pipeline and write the scoreboard.

    Returns the in-memory scoreboard dict.
    """
    log.info("Loading historical events %s — %s (%s)", start, end, sport)
    if sport == "nfl":
        loader = NFLverseHistorical()
        events = loader.fetch_events(start, end, sport="nfl")
        results = loader.load_results(start, end)
    else:
        events, results = [], []
    log.info("Loaded %d events, %d results", len(events), len(results))

    _attach_market_source_prob(events)

    # Blend with current weights (equal by default)
    weights = load_weights()
    blended_events = blend_events(events, weights)

    # Attach signals
    for ev in blended_events:
        chalk = detect_favlong(ev)
        if chalk:
            ev.signals.append(chalk)
        ev.signals.extend(detect_sharp(ev))

    # Per-source scoreboard
    scoreboard = build_scoreboard(blended_events, results)

    # Add a row for the blended model
    bm = _score_blended(blended_events, results)
    if bm:
        scoreboard["flashcat-blended"] = bm

    # Add bankroll curve data
    bankroll = _bankroll_curve(blended_events, results)

    out_path = output_path or SOURCE_SCOREBOARD_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "window": {"start": str(start), "end": str(end), "sport": sport},
        "weights": load_weights(),
        "n_events": len(blended_events),
        "sources": scoreboard,
        "bankroll_curve": bankroll,
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    log.info("Wrote scoreboard → %s", out_path)
    return out_payload


def _score_blended(events: list[Event], results: list[HistoricalResult]) -> dict | None:
    res_by_id = {r.event_id: r for r in results}
    n = 0
    brier_sum = 0.0
    wagered = 0.0
    profit = 0.0
    wins = 0
    losses = 0
    calibration_data: list[tuple[float, bool]] = []
    sliced: dict[str, dict[str, float]] = defaultdict(
        lambda: {"wagered": 0.0, "profit": 0.0, "wins": 0, "losses": 0}
    )
    for ev in events:
        if ev.blended_home_prob is None or ev.pick is None:
            continue
        res = res_by_id.get(ev.event_id)
        if not res:
            continue
        n += 1
        brier_sum += brier_score(ev.blended_home_prob, res.home_won)
        calibration_data.append((ev.blended_home_prob, res.home_won))
        bet = simulate_bet(ev, res, ev.pick, stake=FLAT_STAKE)
        if not bet:
            continue
        wagered += bet.stake
        profit += bet.profit or 0.0
        if bet.won:
            wins += 1
        else:
            losses += 1
        for sig in ev.signals or ["no-signal"]:
            sliced[sig]["wagered"] += bet.stake
            sliced[sig]["profit"] += bet.profit or 0.0
            if bet.won:
                sliced[sig]["wins"] += 1
            else:
                sliced[sig]["losses"] += 1
    if n == 0:
        return None
    from .grader import calibration_bins

    return {
        "n_events": n,
        "brier": brier_sum / n,
        "roi": (profit / wagered) if wagered else None,
        "wagered": wagered,
        "profit": profit,
        "wins": wins,
        "losses": losses,
        "calibration": calibration_bins(calibration_data),
        "slices": {
            k: {**v, "roi": (v["profit"] / v["wagered"]) if v["wagered"] else None}
            for k, v in sliced.items()
        },
    }


def _bankroll_curve(events: list[Event], results: list[HistoricalResult]) -> list[dict]:
    res_by_id = {r.event_id: r for r in results}
    ordered = sorted(
        (e for e in events if e.pick is not None and e.event_id in res_by_id),
        key=lambda e: e.commence_time,
    )
    running = 0.0
    out: list[dict] = []
    for ev in ordered:
        res = res_by_id[ev.event_id]
        bet = simulate_bet(ev, res, ev.pick, stake=FLAT_STAKE)
        if not bet:
            continue
        running += bet.profit or 0.0
        out.append(
            {
                "event_id": ev.event_id,
                "date": ev.commence_time.isoformat(),
                "running_profit": round(running, 2),
            }
        )
    return out
