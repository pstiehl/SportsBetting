"""Flat-$100 simulator for the walk-forward NBA backtest.

Phil's spec: place a hypothetical $100 wager on EVERY model-graded
historical event during backtest. ``qualifying`` means ``the model
produced a prediction``. We explicitly do NOT apply the production
+3pp edge gate NOR the per-sport mode gate, because the point of the
backtest is to honestly evaluate model quality, not to demonstrate
the gate works.

Honesty caveat for closing line / CLV:

source_history.db does NOT contain market_close_decimal for any NBA
row at the time of this writing. We use the prior consensus
(devig'd mean of 538-raptor + 538-elo-modern + bref-srs on the picked
side) as a **market proxy** and convert it to decimal odds with a
standard 4.5% NBA-moneyline hold. This is a known limitation and is
reported in the output table loudly. Real CLV is Phase-2 — pending
The Odds API historical NBA closing-odds backfill.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from .model import FoldResult

log = logging.getLogger(__name__)

DEFAULT_HOLD = 0.045
DEFAULT_STAKE = 100.0


def _prob_to_decimal_with_hold(true_prob: float, hold: float = DEFAULT_HOLD) -> float:
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
    pick_prob: float
    market_implied_prob: float
    market_decimal: float
    won: bool
    pnl: float
    edge_pp: float
    loss_bucket: Optional[str]

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


# NBA loss buckets — per task YAML acceptance criteria.
NBA_LOSS_BUCKETS = (
    "pure_variance",
    "line_moved_against",
    "pace_signal_wrong",
    "rolling_signal_wrong",
    "rest_disadvantage",
    "generic",
)


def _classify_loss_nba(bet: dict, features: dict) -> str:
    """Bucket a losing NBA bet by most-probable narrative cause.

    Priority order (first match wins):

      pure_variance        — model pick prob in [0.45, 0.55]
      line_moved_against   — model edge vs market proxy < 1pp
      rest_disadvantage    — we picked the side with WORSE rest (delta < -1)
                             OR a back-to-back team
      pace_signal_wrong    — prior_dispersion > 0.08 (priors disagreed
                             materially; we picked the wrong side of the
                             dispersion-implied uncertainty)
      rolling_signal_wrong — rolling L10 win% diff > 0.10 in our pick's
                             favor; we still lost
      generic              — fallback
    """
    p = bet["pick_prob"]
    market = bet["market_implied_prob"]
    pick_home = bet["pick_home"]

    if 0.45 <= p <= 0.55:
        return "pure_variance"
    if p - market < 0.01:
        return "line_moved_against"

    rest_diff = features.get("days_rest_diff", 0.0)  # home - away
    b2b_home = features.get("b2b_home", 0.0)
    b2b_away = features.get("b2b_away", 0.0)
    if pick_home and (rest_diff < -1.0 or b2b_home > 0.5):
        return "rest_disadvantage"
    if (not pick_home) and (rest_diff > 1.0 or b2b_away > 0.5):
        return "rest_disadvantage"

    dispersion = features.get("prior_dispersion", 0.0)
    if dispersion > 0.08:
        return "pace_signal_wrong"

    wpct_diff = features.get("win_pct_diff_l10", 0.0)  # home - away
    if pick_home and wpct_diff > 0.10:
        return "rolling_signal_wrong"
    if (not pick_home) and wpct_diff < -0.10:
        return "rolling_signal_wrong"

    return "generic"


def settle_bets(
    folds: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    hold: float = DEFAULT_HOLD,
    edge_gate: Optional[float] = None,
) -> tuple[list[BetResult], dict]:
    """Replay all walk-forward predictions as $100 flat bets.

    Market proxy: prior-consensus (mean of available 538/bref priors) on
    the picked side, converted to decimal odds carrying ``hold``.
    """
    bets: list[BetResult] = []
    for f in folds:
        for pred in f.predictions:
            p = pred["home_prob"]
            pick_home = p >= 0.5
            pick_prob = p if pick_home else 1.0 - p
            # Market proxy: prior consensus on the picked side.
            priors_home = [
                v
                for v in (
                    pred.get("raptor_prob_home"),
                    pred.get("elo_modern_prob_home"),
                    pred.get("bref_srs_prob_home"),
                )
                if v is not None
            ]
            if not priors_home:
                continue
            consensus_home = sum(priors_home) / len(priors_home)
            market_pick_p = consensus_home if pick_home else 1.0 - consensus_home
            edge = pick_prob - market_pick_p
            if edge_gate is not None and edge < edge_gate:
                continue
            market_decimal = _prob_to_decimal_with_hold(market_pick_p, hold=hold)
            won_home = bool(pred["home_won"])
            won = (pick_home and won_home) or (not pick_home and not won_home)
            pnl = stake * (market_decimal - 1.0) if won else -stake
            loss_bucket = (
                None
                if won
                else _classify_loss_nba(
                    {
                        "pick_home": pick_home,
                        "pick_prob": pick_prob,
                        "market_implied_prob": market_pick_p,
                    },
                    pred.get("features", {}),
                )
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
    if not bets:
        return {"overall": _empty_block(), "per_year": {}, "loss_buckets": {}, "n_bets": 0}
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

    clv = sum(b.edge_pp for b in bets) / len(bets)
    avg_market_implied = sum(b.market_implied_prob for b in bets) / len(bets)
    overall["clv_proxy_pp"] = clv
    overall["avg_market_implied_prob"] = avg_market_implied

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
    peak = 0.0
    curve = 0.0
    max_dd = 0.0
    pnl_series: list[float] = []
    for b in sorted(bets, key=lambda x: x.game_date):
        curve += b.pnl
        peak = max(peak, curve)
        max_dd = min(max_dd, curve - peak)
        pnl_series.append(b.pnl)
    # NBA: 82-game regular season; coarse Sharpe annualization.
    if len(pnl_series) > 1:
        mean = sum(pnl_series) / len(pnl_series)
        var = sum((x - mean) ** 2 for x in pnl_series) / max(1, len(pnl_series) - 1)
        sd = math.sqrt(var)
        sharpe = (mean / sd) * math.sqrt(82) if sd > 0 else None
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
    lines: list[str] = []
    lines.append(
        f"Walk-forward NBA backtest — market proxy = prior consensus × (1 + {hold:.2%} hold)"
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
        lines.append(f"CLV proxy (mean edge over prior-consensus): {clv*100:+.2f} pp")
        ami = o.get("avg_market_implied_prob")
        if ami is not None:
            lines.append(f"Mean prior-consensus implied prob on picked side: {ami*100:.2f}%")
    lb = summary.get("loss_buckets") or {}
    if lb:
        lines.append("")
        lines.append("Loss post-mortem (count of LOSING bets by cause):")
        total_losses = sum(lb.values())
        for bucket, n in sorted(lb.items(), key=lambda kv: -kv[1]):
            pct = n / total_losses * 100 if total_losses else 0
            lines.append(f"  {bucket:<25} {n:>6} ({pct:5.1f}%)")
    return "\n".join(lines)
