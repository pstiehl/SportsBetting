"""Flat-$100 simulator for the walk-forward MLB backtest.

Phil's spec: place a hypothetical $100 wager on EVERY qualifying historical
event during backtest. ``qualifying`` means ``the model produced a
prediction`` — we explicitly do NOT apply the production +3pp edge gate
NOR the per-sport mode gate, because the point of the backtest is to
honestly evaluate model quality, not to demonstrate the gate works.

Honesty caveat for closing line / CLV:

The free 538 archive does NOT provide closing odds. We use ``rating_prob_home``
(538's pitcher-adjusted win prob) as a **market proxy** and convert it to
decimal odds with a standard -110/-110-equivalent vig (4.5% hold). This
is a known limitation and is reported in the output table loudly. For a
real CLV signal in the live ledger we'll capture actual closing decimal
odds via the existing market-close source.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .model import FoldResult

log = logging.getLogger(__name__)

# Sportsbook hold to apply to the 538 rating_prob → decimal odds conversion.
# 4.5% is the median two-way moneyline hold across mainstream US books
# (DraftKings, FanDuel, Caesars) for major-league baseball.
DEFAULT_HOLD = 0.045
DEFAULT_STAKE = 100.0


def _prob_to_decimal_with_hold(true_prob: float, hold: float = DEFAULT_HOLD) -> float:
    """Convert a "true" probability to decimal odds carrying ``hold`` vig.

    A two-way market with hold h has overround 1+h. Each side's implied
    probability = true_prob * (1+h). Decimal odds = 1 / implied.
    """
    p = max(0.001, min(0.999, true_prob))
    implied = p * (1.0 + hold)
    implied = min(0.999, implied)
    return 1.0 / implied


@dataclass
class BetResult:
    game_date: str
    home: str
    away: str
    pick_home: bool
    pick_prob: float            # model probability on the picked side
    market_implied_prob: float  # closing implied prob on picked side (proxy)
    market_decimal: float       # closing decimal on picked side (proxy)
    won: bool
    pnl: float                  # net P&L on $100 stake
    edge_pp: float              # pick_prob - market_implied_prob (proxy CLV)
    loss_bucket: str | None     # e.g. 'pure_variance' for losing bets

    def as_row(self) -> dict:
        return {
            "game_date": self.game_date,
            "home": self.home,
            "away": self.away,
            "pick_home": self.pick_home,
            "pick_prob": self.pick_prob,
            "market_implied_prob": self.market_implied_prob,
            "market_decimal": self.market_decimal,
            "won": self.won,
            "pnl": self.pnl,
            "edge_pp": self.edge_pp,
            "loss_bucket": self.loss_bucket,
        }


def _classify_loss(bet: dict, features: dict) -> str:
    """Bucket a losing bet by the most-probable narrative cause.

    Buckets (in priority order):

      pure_variance        — model pick prob within 5pp of outcome 0/1 baseline
                             (i.e. coinflip-ish; calling 51/49 on losing side
                             is variance, not a model failure)
      line_moved_against   — model edge vs market < 1pp (we lost to vig)
      pitcher_signal_wrong — pitcher rating diff strongly favored our pick;
                             pitcher_rgs_diff sign agreed with pick
      rolling_signal_wrong — rolling L10 run-diff sign agreed with pick by
                             > 1 run/game; we still lost
      generic              — fallback when none of the above triggers
    """
    p = bet["pick_prob"]
    market = bet["market_implied_prob"]
    pick_home = bet["pick_home"]

    if 0.45 <= p <= 0.55:
        return "pure_variance"
    edge = p - market
    if edge < 0.01:
        return "line_moved_against"
    pr_diff = (features or {}).get("pitcher_rgs_diff")
    if pr_diff is not None:
        # If we picked home and pitcher rgs diff was strongly + (favoring
        # home), or picked away and rgs diff strongly - (favoring away),
        # then the pitcher signal agreed with us and we lost anyway.
        agreed = (pick_home and pr_diff > 0.5) or (not pick_home and pr_diff < -0.5)
        if agreed:
            return "pitcher_signal_wrong"
    rd_diff = (features or {}).get("run_diff_l10_diff")
    if rd_diff is not None:
        agreed = (pick_home and rd_diff > 1.0) or (not pick_home and rd_diff < -1.0)
        if agreed:
            return "rolling_signal_wrong"
    return "generic"


def simulate(
    folds: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    hold: float = DEFAULT_HOLD,
    edge_gate: float | None = None,
) -> tuple[list[BetResult], dict]:
    """Run the flat-stake simulator on the walk-forward fold predictions.

    ``edge_gate`` is normally ``None`` (Phil's spec — bet every game),
    but exposed so we can compute "what would the production +3pp gate
    have caught" as a side metric.

    Returns ``(bets, summary)`` where ``summary`` is a per-year + total
    aggregate.
    """
    bets: list[BetResult] = []
    for f in folds:
        for pred in f.predictions:
            p = pred["home_prob"]
            pick_home = p >= 0.5
            pick_prob = p if pick_home else 1.0 - p
            # Market proxy: 538 rating_prob on the picked side; fall back
            # to elo_prob if rating_prob is unavailable.
            rating_p = pred.get("rating_prob_home")
            if rating_p is None:
                rating_p = pred.get("elo_prob_home")
            if rating_p is None:
                # Without a market proxy we can't simulate this bet.
                continue
            market_pick_p = rating_p if pick_home else 1.0 - rating_p
            edge = pick_prob - market_pick_p
            if edge_gate is not None and edge < edge_gate:
                continue
            market_decimal = _prob_to_decimal_with_hold(market_pick_p, hold=hold)
            won_home = bool(pred["home_won"])
            won = (pick_home and won_home) or (not pick_home and not won_home)
            pnl = stake * (market_decimal - 1.0) if won else -stake
            loss_bucket = None if won else _classify_loss(
                {"pick_home": pick_home, "pick_prob": pick_prob,
                 "market_implied_prob": market_pick_p},
                pred.get("features", {}),
            )
            bets.append(BetResult(
                game_date=pred["game_date"],
                home=pred["home"],
                away=pred["away"],
                pick_home=pick_home,
                pick_prob=pick_prob,
                market_implied_prob=market_pick_p,
                market_decimal=market_decimal,
                won=won,
                pnl=pnl,
                edge_pp=edge,
                loss_bucket=loss_bucket,
            ))

    summary = summarize(bets, stake=stake)
    return bets, summary


def summarize(bets: list[BetResult], *, stake: float = DEFAULT_STAKE) -> dict:
    """Aggregate metrics overall and per-year. Includes drawdown & Sharpe."""
    if not bets:
        return {"overall": _empty_block(), "per_year": {}, "loss_buckets": {}}
    by_year: dict[int, list[BetResult]] = defaultdict(list)
    for b in bets:
        try:
            yr = int(b.game_date[:4])
        except (ValueError, TypeError):
            continue
        by_year[yr].append(b)

    overall = _block_metrics(bets, stake=stake)
    per_year: dict[int, dict] = {}
    for yr in sorted(by_year):
        per_year[yr] = _block_metrics(by_year[yr], stake=stake)

    # CLV proxy summary: mean edge_pp on placed bets.
    clv = sum(b.edge_pp for b in bets) / len(bets)
    avg_market_implied = sum(b.market_implied_prob for b in bets) / len(bets)
    overall["clv_proxy_pp"] = clv
    overall["avg_market_implied_prob"] = avg_market_implied

    # Loss bucket aggregate.
    buckets: dict[str, int] = defaultdict(int)
    for b in bets:
        if not b.won and b.loss_bucket:
            buckets[b.loss_bucket] += 1
    return {
        "overall": overall,
        "per_year": per_year,
        "loss_buckets": dict(buckets),
        "n_bets": len(bets),
    }


def _empty_block() -> dict:
    return {
        "n_bets": 0, "n_wins": 0, "win_rate": None,
        "wagered": 0.0, "profit": 0.0, "roi": None,
        "max_drawdown": 0.0, "sharpe": None,
    }


def _block_metrics(bets: list[BetResult], *, stake: float) -> dict:
    n = len(bets)
    if n == 0:
        return _empty_block()
    wins = sum(1 for b in bets if b.won)
    wagered = n * stake
    profit = sum(b.pnl for b in bets)
    # Running drawdown
    peak = 0.0
    curve = 0.0
    max_dd = 0.0
    pnl_series: list[float] = []
    for b in sorted(bets, key=lambda x: x.game_date):
        curve += b.pnl
        peak = max(peak, curve)
        max_dd = min(max_dd, curve - peak)
        pnl_series.append(b.pnl)
    # Annualized Sharpe assuming 162-game season day count (Phil's spec).
    if len(pnl_series) > 1:
        mean = sum(pnl_series) / len(pnl_series)
        var = sum((x - mean) ** 2 for x in pnl_series) / max(1, len(pnl_series) - 1)
        sd = math.sqrt(var)
        # Sharpe per-bet → annualized by sqrt(162) (treating a season as 162
        # bet days when fully wagered; this is a coarse but standard proxy).
        sharpe = (mean / sd) * math.sqrt(162) if sd > 0 else None
    else:
        sharpe = None
    return {
        "n_bets": n,
        "n_wins": wins,
        "win_rate": wins / n,
        "wagered": round(wagered, 2),
        "profit": round(profit, 2),
        "roi": profit / wagered if wagered > 0 else None,
        "max_drawdown": round(max_dd, 2),
        "sharpe": sharpe,
    }


def format_summary_table(summary: dict, *, hold: float = DEFAULT_HOLD) -> str:
    """Render the headline metrics table for the PR body / status doc."""
    lines: list[str] = []
    lines.append(
        f"Walk-forward MLB backtest — market proxy = 538 rating_prob × (1 + {hold:.2%} hold)"
    )
    lines.append("")
    header = (
        f"{'Year':<7} {'n_bets':>7} {'wins':>6} {'win%':>7} "
        f"{'wagered':>10} {'profit':>10} {'ROI':>8} {'max_dd':>9} {'Sharpe':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    py = summary.get("per_year") or {}
    for yr in sorted(py):
        b = py[yr]
        roi_s = f"{b['roi']*100:+.2f}%" if b['roi'] is not None else "n/a"
        wr_s = f"{b['win_rate']*100:.1f}%" if b['win_rate'] is not None else "n/a"
        sh_s = f"{b['sharpe']:+.2f}" if b['sharpe'] is not None else "n/a"
        lines.append(
            f"{yr:<7} {b['n_bets']:>7} {b['n_wins']:>6} {wr_s:>7} "
            f"${b['wagered']:>8,.0f} ${b['profit']:>+8,.0f} {roi_s:>8} "
            f"${b['max_drawdown']:>7,.0f} {sh_s:>8}"
        )
    o = summary.get("overall") or {}
    if o.get("n_bets"):
        lines.append("-" * len(header))
        roi_s = f"{o['roi']*100:+.2f}%" if o['roi'] is not None else "n/a"
        wr_s = f"{o['win_rate']*100:.1f}%" if o['win_rate'] is not None else "n/a"
        sh_s = f"{o['sharpe']:+.2f}" if o['sharpe'] is not None else "n/a"
        lines.append(
            f"{'TOTAL':<7} {o['n_bets']:>7} {o['n_wins']:>6} {wr_s:>7} "
            f"${o['wagered']:>8,.0f} ${o['profit']:>+8,.0f} {roi_s:>8} "
            f"${o['max_drawdown']:>7,.0f} {sh_s:>8}"
        )
    clv = o.get("clv_proxy_pp")
    if clv is not None:
        lines.append("")
        lines.append(f"CLV proxy (mean edge over market proxy): {clv*100:+.2f} pp")
        ami = o.get("avg_market_implied_prob")
        if ami is not None:
            lines.append(f"Mean market implied prob on picked side: {ami*100:.2f}%")
    lb = summary.get("loss_buckets") or {}
    if lb:
        lines.append("")
        lines.append("Loss post-mortem (count of LOSING bets by cause):")
        total_losses = sum(lb.values())
        for bucket, n in sorted(lb.items(), key=lambda kv: -kv[1]):
            pct = n / total_losses * 100 if total_losses else 0
            lines.append(f"  {bucket:<25} {n:>6} ({pct:5.1f}%)")
    return "\n".join(lines)
