"""Brier score, ROI, calibration math + per-source scoreboard."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..config import FLAT_STAKE
from ..types import (
    Bet,
    Event,
    HistoricalResult,
    Side,
    american_to_profit,
    american_to_prob,
)


def brier_score(prob: float, outcome: bool) -> float:
    """Brier score for a single binary forecast. Lower = better. Always in [0,1]."""
    y = 1.0 if outcome else 0.0
    return (prob - y) ** 2


def _book_avg_price(event: Event, side: Side) -> int | None:
    prices = [ln.american for ln in event.lines if ln.side == side and not ln.is_opening]
    if not prices:
        prices = [ln.american for ln in event.lines if ln.side == side]
    if not prices:
        return None
    # Median price is the right summary; we'll use rounded mean to keep math obvious.
    return int(round(sum(prices) / len(prices)))


def simulate_bet(
    event: Event,
    result: HistoricalResult,
    side: Side,
    stake: float = FLAT_STAKE,
) -> Bet | None:
    """Simulate a $100 bet on `side` at the book-average price for that side."""
    price = _book_avg_price(event, side)
    if price is None:
        return None
    home_won = result.home_won
    won = (side == "home" and home_won) or (side == "away" and not home_won)
    profit = american_to_profit(price, stake) if won else -stake
    return Bet(
        event_id=event.event_id,
        side=side,
        stake=stake,
        american_price=price,
        won=won,
        profit=profit,
    )


def simulate_bets(
    events: list[Event],
    results: list[HistoricalResult],
    pick_fn,
    stake: float = FLAT_STAKE,
) -> list[Bet]:
    """Simulate bets using a pick function `pick_fn(event) -> (side, prob)`."""
    res_by_id = {r.event_id: r for r in results}
    out: list[Bet] = []
    for ev in events:
        res = res_by_id.get(ev.event_id)
        if not res:
            continue
        pick = pick_fn(ev)
        if pick is None:
            continue
        side, _prob = pick
        bet = simulate_bet(ev, res, side, stake=stake)
        if bet:
            out.append(bet)
    return out


def calibration_bins(
    forecasts: Iterable[tuple[float, bool]], n_bins: int = 10
) -> list[dict]:
    """Bin predicted probabilities and compute the empirical hit rate per bin."""
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for p, y in forecasts:
        b = min(n_bins - 1, max(0, int(p * n_bins)))
        buckets[b].append((p, y))
    out: list[dict] = []
    for b in range(n_bins):
        rows = buckets.get(b, [])
        if not rows:
            out.append(
                {
                    "bin": b,
                    "lo": b / n_bins,
                    "hi": (b + 1) / n_bins,
                    "n": 0,
                    "predicted": None,
                    "actual": None,
                }
            )
            continue
        mean_p = sum(p for p, _ in rows) / len(rows)
        hit = sum(1 for _, y in rows if y) / len(rows)
        out.append(
            {
                "bin": b,
                "lo": b / n_bins,
                "hi": (b + 1) / n_bins,
                "n": len(rows),
                "predicted": mean_p,
                "actual": hit,
            }
        )
    return out


def build_scoreboard(
    events: list[Event],
    results: list[HistoricalResult],
) -> dict[str, dict]:
    """Per-source Brier + ROI on $100-flat using that source's prob alone.

    Each source picks the side with higher implied home prob (or away if <0.5),
    and we record the realized P&L using book-average prices.
    """
    res_by_id = {r.event_id: r for r in results}
    rows: dict[str, dict] = defaultdict(
        lambda: {
            "n_events": 0,
            "brier_sum": 0.0,
            "wagered": 0.0,
            "profit": 0.0,
            "wins": 0,
            "losses": 0,
            "calibration_data": [],
        }
    )
    for ev in events:
        res = res_by_id.get(ev.event_id)
        if not res:
            continue
        for sp in ev.source_probs:
            home_won = res.home_won
            rows[sp.source]["n_events"] += 1
            rows[sp.source]["brier_sum"] += brier_score(sp.home_win_prob, home_won)
            rows[sp.source]["calibration_data"].append((sp.home_win_prob, home_won))
            side: Side = "home" if sp.home_win_prob >= 0.5 else "away"
            bet = simulate_bet(ev, res, side)
            if not bet:
                continue
            rows[sp.source]["wagered"] += bet.stake
            rows[sp.source]["profit"] += bet.profit or 0.0
            if bet.won:
                rows[sp.source]["wins"] += 1
            else:
                rows[sp.source]["losses"] += 1
        # Also score the market itself as a "source" via devigged consensus
        if ev.lines:
            for side_label in ("home", "away"):
                pass
            home_imp = _consensus_home_prob(ev)
            if home_imp is not None:
                rows["market-consensus"]["n_events"] += 1
                rows["market-consensus"]["brier_sum"] += brier_score(home_imp, res.home_won)
                rows["market-consensus"]["calibration_data"].append((home_imp, res.home_won))
                side = "home" if home_imp >= 0.5 else "away"
                bet = simulate_bet(ev, res, side)
                if bet:
                    rows["market-consensus"]["wagered"] += bet.stake
                    rows["market-consensus"]["profit"] += bet.profit or 0.0
                    if bet.won:
                        rows["market-consensus"]["wins"] += 1
                    else:
                        rows["market-consensus"]["losses"] += 1
    out: dict[str, dict] = {}
    for src, r in rows.items():
        n = r["n_events"]
        out[src] = {
            "n_events": n,
            "brier": (r["brier_sum"] / n) if n else None,
            "roi": (r["profit"] / r["wagered"]) if r["wagered"] else None,
            "wagered": r["wagered"],
            "profit": r["profit"],
            "wins": r["wins"],
            "losses": r["losses"],
            "calibration": calibration_bins(r["calibration_data"]),
        }
    return out


def _consensus_home_prob(event: Event) -> float | None:
    from ..types import devig_two_way

    home_prices = [
        american_to_prob(ln.american)
        for ln in event.lines
        if ln.side == "home" and not ln.is_opening
    ]
    away_prices = [
        american_to_prob(ln.american)
        for ln in event.lines
        if ln.side == "away" and not ln.is_opening
    ]
    if not home_prices or not away_prices:
        return None
    h = sum(home_prices) / len(home_prices)
    a = sum(away_prices) / len(away_prices)
    h_devig, _ = devig_two_way(h, a)
    return h_devig
