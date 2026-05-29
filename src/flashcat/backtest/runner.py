"""Backtest runner — pulls historical data, blends, simulates, writes scoreboard.

Two entry points:
  - ``run_backtest(start, end, sport)`` — single sport
  - ``run_multi_sport_backtest(start, end, sports=...)`` — per-sport blocks
"""

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
    edge_threshold,
    stake_mode,
)
from ..model.blend import blend_events, load_weights
from ..model.staking import decide_stake
from ..signals.favlong import detect as detect_favlong
from ..signals.sharp import detect as detect_sharp
from ..sources.nflverse import NFLverseHistorical
from ..sources.tennis_history import TennisDataHistorical
from ..sources.nba_history import FiveThirtyEightNBAHistorical
from ..sources.fivethirtyeight_archives import (
    FiveThirtyEightMLBElo,
    FiveThirtyEightNBAModern,
    FiveThirtyEightNFLElo,
)
from ..types import (
    Event,
    HistoricalResult,
    SourceProb,
    Side,
    american_to_prob,
    devig_two_way,
)
from .grader import brier_score, build_scoreboard, simulate_bet, _book_avg_price

log = logging.getLogger(__name__)

# Per-sport list of historical connectors. Each backtested sport may have
# multiple connectors; we merge their events on (date, home, away).
SPORT_LOADERS: dict[str, list] = {
    "nfl": [NFLverseHistorical, FiveThirtyEightNFLElo],
    "atp": [lambda: TennisDataHistorical(tour="atp")],
    "wta": [lambda: TennisDataHistorical(tour="wta")],
    "nba": [FiveThirtyEightNBAHistorical, FiveThirtyEightNBAModern],
    "mlb": [FiveThirtyEightMLBElo],
}


def _normalize_team(name: str) -> str:
    return (name or "").lower().replace(".", "").strip()


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


def _load_for_sport(sport: str, start: date, end: date) -> tuple[list[Event], list[HistoricalResult]]:
    """Load historical events + results for one sport from every wired connector.

    Events from multiple connectors are merged on a (date, sorted-team-pair)
    key so the blender sees one Event per game with N source_probs attached.
    """
    loaders = SPORT_LOADERS.get(sport, [])
    if not loaders:
        log.warning("no historical loader for sport=%s — skipping", sport)
        return [], []

    merged_events: dict[tuple, Event] = {}
    merged_results: dict[str, HistoricalResult] = {}

    for loader_factory in loaders:
        try:
            loader = loader_factory() if callable(loader_factory) else loader_factory
            ev_list = loader.fetch_events(start, end, sport=sport)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            log.warning("%s loader fetch_events failed: %s", sport, e)
            ev_list = []
        try:
            res_list = loader.load_results(start, end) if hasattr(loader, "load_results") else []
        except Exception as e:  # noqa: BLE001
            log.warning("%s loader load_results failed: %s", sport, e)
            res_list = []

        for ev in ev_list:
            key = _merge_key(ev)
            if key in merged_events:
                target = merged_events[key]
                target.source_probs.extend(ev.source_probs)
                target.lines.extend(ev.lines)
            else:
                merged_events[key] = ev

        # Map results onto the merged event_ids (use whichever loader
        # produced the first event for the matchup).
        for res in res_list:
            key = (
                res.commence_time.date().isoformat(),
                *sorted([_normalize_team(res.home), _normalize_team(res.away)]),
            )
            ev = merged_events.get(key)
            if ev is None:
                # Result without an event — keep with its original id.
                merged_results.setdefault(res.event_id, res)
                continue
            # Re-key the result to the merged event's id, but only if the
            # result corresponds to the same home/away orientation.
            same_home = _normalize_team(res.home) == _normalize_team(ev.home)
            if same_home:
                merged_results.setdefault(
                    ev.event_id,
                    HistoricalResult(
                        event_id=ev.event_id,
                        sport=res.sport,
                        home=ev.home,
                        away=ev.away,
                        commence_time=res.commence_time,
                        home_won=res.home_won,
                        home_score=res.home_score,
                        away_score=res.away_score,
                    ),
                )
            else:
                merged_results.setdefault(
                    ev.event_id,
                    HistoricalResult(
                        event_id=ev.event_id,
                        sport=res.sport,
                        home=ev.home,
                        away=ev.away,
                        commence_time=res.commence_time,
                        home_won=not res.home_won,
                        home_score=res.away_score,
                        away_score=res.home_score,
                    ),
                )

    return list(merged_events.values()), list(merged_results.values())


def _merge_key(ev: Event) -> tuple:
    return (
        ev.commence_time.date().isoformat(),
        *sorted([_normalize_team(ev.home), _normalize_team(ev.away)]),
    )


def run_backtest(
    start: date,
    end: date,
    sport: str = "nfl",
    output_path: Path | None = None,
) -> dict:
    """Run the full backtest pipeline for a single sport and write the scoreboard."""
    log.info("Loading historical events %s — %s (%s)", start, end, sport)
    events, results = _load_for_sport(sport, start, end)
    log.info("Loaded %d events, %d results", len(events), len(results))

    _attach_market_source_prob(events)

    weights = load_weights()
    blended_events = blend_events(events, weights)

    for ev in blended_events:
        chalk = detect_favlong(ev)
        if chalk:
            ev.signals.append(chalk)
        ev.signals.extend(detect_sharp(ev))

    scoreboard = build_scoreboard(blended_events, results)

    bm = _score_blended(blended_events, results, sport=sport)
    if bm:
        scoreboard["flashcat-blended"] = bm

    bankroll = _bankroll_curve(blended_events, results)

    out_path = output_path or SOURCE_SCOREBOARD_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "window": {"start": str(start), "end": str(end), "sport": sport},
        "weights": load_weights(),
        "n_events": len(blended_events),
        "sources": scoreboard,
        "bankroll_curve": bankroll,
        "stake_mode": stake_mode(),
        "edge_threshold": edge_threshold(),
    }
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    log.info("Wrote scoreboard → %s", out_path)
    return out_payload


def run_multi_sport_backtest(
    start: date,
    end: date,
    sports: list[str] | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run the backtest separately for each sport and write a combined scoreboard.

    The top-level ``sources`` block is keyed by ``"<sport>:<source>"`` so the
    site can show per-sport ROI per source.
    """
    sports = sports or ["nfl", "nba", "mlb", "atp", "wta"]
    log.info("Multi-sport backtest %s — %s, sports=%s", start, end, sports)

    weights = load_weights()
    combined_sources: dict[str, dict] = {}
    per_sport: dict[str, dict] = {}
    combined_bankroll: list[dict] = []
    running = 0.0

    for sport in sports:
        events, results = _load_for_sport(sport, start, end)
        log.info("  %s: %d events, %d results", sport, len(events), len(results))
        if not events:
            per_sport[sport] = {"n_events": 0, "blended": None, "sources": {}}
            continue
        _attach_market_source_prob(events)
        blended = blend_events(events, weights)
        for ev in blended:
            chalk = detect_favlong(ev)
            if chalk:
                ev.signals.append(chalk)
            ev.signals.extend(detect_sharp(ev))
        sport_sb = build_scoreboard(blended, results)
        bm = _score_blended(blended, results, sport=sport)
        per_sport[sport] = {
            "n_events": len(blended),
            "sources": sport_sb,
            "blended": bm,
        }
        for src, row in sport_sb.items():
            combined_sources[f"{sport}:{src}"] = row
        # Continue bankroll curve across sports for one global trajectory.
        sport_bankroll = _bankroll_curve(blended, results, start_balance=running)
        if sport_bankroll:
            running = sport_bankroll[-1]["running_profit"]
        combined_bankroll.extend(sport_bankroll)

    blended_overall = _aggregate_blended(per_sport)

    out_payload = {
        "window": {"start": str(start), "end": str(end), "sport": "multi"},
        "weights": weights,
        "n_events": sum(p["n_events"] for p in per_sport.values()),
        "sources": combined_sources,
        "per_sport": per_sport,
        "blended_overall": blended_overall,
        "bankroll_curve": combined_bankroll,
        "stake_mode": stake_mode(),
        "edge_threshold": edge_threshold(),
    }
    out_path = output_path or SOURCE_SCOREBOARD_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    log.info("Wrote multi-sport scoreboard → %s", out_path)
    return out_payload


def _aggregate_blended(per_sport: dict[str, dict]) -> dict:
    """Aggregate blended ROI across sports (weighted by wagered)."""
    n_events = 0
    wagered = 0.0
    profit = 0.0
    wins = 0
    losses = 0
    for p in per_sport.values():
        bm = p.get("blended")
        if not bm:
            continue
        n_events += bm.get("n_events", 0)
        wagered += bm.get("wagered", 0.0) or 0.0
        profit += bm.get("profit", 0.0) or 0.0
        wins += bm.get("wins", 0) or 0
        losses += bm.get("losses", 0) or 0
    return {
        "n_events": n_events,
        "wagered": wagered,
        "profit": profit,
        "wins": wins,
        "losses": losses,
        "roi": (profit / wagered) if wagered else None,
    }


def _score_blended(events: list[Event], results: list[HistoricalResult], sport: str = "") -> dict | None:
    """Score the blended model using the configured Kelly/edge staking rule."""
    res_by_id = {r.event_id: r for r in results}
    n = 0
    brier_sum = 0.0
    wagered = 0.0
    profit = 0.0
    wins = 0
    losses = 0
    skipped = 0
    skip_reasons: dict[str, int] = defaultdict(int)
    calibration_data: list[tuple[float, bool]] = []
    sliced: dict[str, dict[str, float]] = defaultdict(
        lambda: {"wagered": 0.0, "profit": 0.0, "wins": 0, "losses": 0, "n": 0}
    )
    mode = stake_mode()
    thresh = edge_threshold()

    for ev in events:
        if ev.blended_home_prob is None or ev.pick is None:
            continue
        res = res_by_id.get(ev.event_id)
        if not res:
            continue
        n += 1
        brier_sum += brier_score(ev.blended_home_prob, res.home_won)
        calibration_data.append((ev.blended_home_prob, res.home_won))

        decision = decide_stake(
            ev, ev.pick, ev.pick_prob or 0.5,
            mode=mode, edge_threshold=thresh,
        )
        if decision.stake <= 0:
            skipped += 1
            skip_reasons[decision.skipped_reason or "unknown"] += 1
            continue
        # Use the same simulate_bet for grading consistency, but override stake.
        bet = simulate_bet(ev, res, ev.pick, stake=decision.stake)
        if not bet:
            skipped += 1
            skip_reasons["no_market_price"] += 1
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
            sliced[sig]["n"] += 1
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
        "skipped": skipped,
        "skip_reasons": dict(skip_reasons),
        "stake_mode": mode,
        "edge_threshold": thresh,
        "calibration": calibration_bins(calibration_data),
        "slices": {
            k: {**v, "roi": (v["profit"] / v["wagered"]) if v["wagered"] else None}
            for k, v in sliced.items()
        },
    }


def _bankroll_curve(events: list[Event], results: list[HistoricalResult], start_balance: float = 0.0) -> list[dict]:
    res_by_id = {r.event_id: r for r in results}
    ordered = sorted(
        (e for e in events if e.pick is not None and e.event_id in res_by_id),
        key=lambda e: e.commence_time,
    )
    running = start_balance
    out: list[dict] = []
    mode = stake_mode()
    thresh = edge_threshold()
    for ev in ordered:
        res = res_by_id[ev.event_id]
        decision = decide_stake(
            ev, ev.pick, ev.pick_prob or 0.5,
            mode=mode, edge_threshold=thresh,
        )
        if decision.stake <= 0:
            continue
        bet = simulate_bet(ev, res, ev.pick, stake=decision.stake)
        if not bet:
            continue
        running += bet.profit or 0.0
        out.append(
            {
                "event_id": ev.event_id,
                "sport": ev.sport,
                "date": ev.commence_time.isoformat(),
                "running_profit": round(running, 2),
            }
        )
    return out
