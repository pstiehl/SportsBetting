"""Flat-$100 simulator for the walk-forward NBA backtest — Phase 2.

Phil's spec: place a hypothetical $100 wager on EVERY qualifying historical
game during backtest. No production edge gate applied in the base run.

Market proxy:
  No NBA moneyline historical archive is freely available (sportsbookreview
  dead; Odds API requires paid key). We use the bref SRS prior probability
  (loaded from source_history.db) as the market proxy with a 4.5% hold.
  CLV is therefore a proxy (model prob − srs_prior_prob) not true CLV.

Loss buckets (NBA adaptation):
  pure_variance       — model prob in [0.45, 0.55]; coinflip
  line_moved_against  — model edge over proxy < 1pp; lost to vig
  form_signal_wrong   — rolling pt_diff_l5_diff strongly favored our pick;
                        we lost (NBA equivalent of rolling_signal_wrong)
  b2b_fatigue         — picked a team on a back-to-back; they lost
  generic             — none of the above
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .model import FoldResult

log = logging.getLogger(__name__)

DEFAULT_HOLD = 0.045
DEFAULT_STAKE = 100.0
NBA_HFA_SIGMA = 11.0


def _prob_to_decimal(true_prob: float, hold: float = DEFAULT_HOLD) -> float:
    """True probability → decimal odds with hold vig."""
    p = max(0.001, min(0.999, true_prob))
    implied = p * (1.0 + hold)
    implied = min(0.999, implied)
    return 1.0 / implied


def _classify_loss(bet: dict, features: dict) -> str:
    """Bucket a losing bet by most-probable narrative cause."""
    p = bet["pick_prob"]
    market = bet["market_implied_prob"]
    pick_home = bet["pick_home"]

    # 1. Coinflip
    if 0.45 <= p <= 0.55:
        return "pure_variance"

    # 2. Lost to vig — edge was < 1pp
    if abs(p - market) < 0.01:
        return "line_moved_against"

    # 3. Form signal wrong — rolling L5 pt diff strongly backed our pick
    pt_diff = features.get("pt_diff_l5_diff")  # positive = home advantaged
    if pt_diff is not None and abs(pt_diff) > 2.0:
        signal_home = pt_diff > 0
        if (signal_home and pick_home) or (not signal_home and not pick_home):
            return "form_signal_wrong"

    # 4. B2B fatigue — we picked the B2B team
    b2b_home = features.get("b2b_home", 0.0) or 0.0
    b2b_away = features.get("b2b_away", 0.0) or 0.0
    if pick_home and b2b_home > 0.5:
        return "b2b_fatigue"
    if not pick_home and b2b_away > 0.5:
        return "b2b_fatigue"

    return "generic"


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


def simulate(
    folds: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    hold: float = DEFAULT_HOLD,
    edge_gate: Optional[float] = None,
) -> tuple[list[BetResult], dict]:
    """Flat-stake sim on all walk-forward predictions.

    ``edge_gate`` — if not None, only bet when model prob − market implied > gate.
    """
    bets: list[BetResult] = []
    per_year: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0}
    )
    loss_buckets: dict[str, int] = defaultdict(int)

    for fold in folds:
        for pred in fold.predictions:
            home_prob = pred["home_prob"]
            away_prob = 1.0 - home_prob
            home_won = pred["home_won"]

            # Market proxy from SRS prior
            prior = pred.get("prior_prob_home")
            if prior is None:
                # No SRS prior — use flat 0.55 HFA placeholder
                prior = 0.55

            features = pred.get("features", {}) or {}

            # Pick the side the model favors
            pick_home = home_prob >= 0.5
            pick_prob = home_prob if pick_home else away_prob
            market_pick_prob = prior if pick_home else (1.0 - prior)

            market_decimal = _prob_to_decimal(market_pick_prob, hold)
            market_implied = 1.0 / market_decimal

            edge_pp = pick_prob - market_implied
            if edge_gate is not None and edge_pp < edge_gate:
                continue

            won = (home_won and pick_home) or (not home_won and not pick_home)
            pnl = (market_decimal - 1.0) * stake if won else -stake

            bucket: Optional[str] = None
            if not won:
                bucket = _classify_loss(
                    {
                        "pick_prob": pick_prob,
                        "market_implied_prob": market_implied,
                        "pick_home": pick_home,
                    },
                    features,
                )
                loss_buckets[bucket] += 1

            yr = int(pred["game_date"][:4])
            per_year[yr]["n"] += 1
            per_year[yr]["wins"] += int(won)
            per_year[yr]["pnl"] += pnl

            bets.append(
                BetResult(
                    game_date=pred["game_date"],
                    home=pred["home"],
                    away=pred["away"],
                    pick_home=pick_home,
                    pick_prob=pick_prob,
                    market_implied_prob=market_implied,
                    market_decimal=market_decimal,
                    won=won,
                    pnl=pnl,
                    edge_pp=edge_pp,
                    loss_bucket=bucket,
                )
            )

    n = len(bets)
    wins = sum(1 for b in bets if b.won)
    total_pnl = sum(b.pnl for b in bets)
    total_stake = n * stake
    roi = total_pnl / total_stake if total_stake > 0 else None

    # CLV proxy
    clv_vals = [b.edge_pp for b in bets]
    clv_proxy = sum(clv_vals) / len(clv_vals) if clv_vals else None

    # Max drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for b in sorted(bets, key=lambda x: x.game_date):
        cum += b.pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized to ~82-game NBA regular season per team)
    if bets:
        pnl_list = [b.pnl for b in bets]
        mu = sum(pnl_list) / len(pnl_list)
        variance = sum((p - mu) ** 2 for p in pnl_list) / max(len(pnl_list) - 1, 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        # ~1230 games/season, ~41 team-game pairs per day, normalize to 82 "seasons"
        ann_factor = math.sqrt(1230)
        sharpe = (mu / std) * ann_factor if std > 0 else None
    else:
        sharpe = None

    # Loss bucket %
    total_losses = sum(loss_buckets.values())
    bucket_pct = {
        k: round(v / total_losses * 100, 1) if total_losses > 0 else 0.0
        for k, v in loss_buckets.items()
    }

    per_year_out = {}
    for yr, d in sorted(per_year.items()):
        nn = d["n"]
        roi_yr = d["pnl"] / (nn * stake) if nn > 0 else None
        per_year_out[str(yr)] = {
            "n_bets": nn,
            "win_rate": d["wins"] / nn if nn > 0 else None,
            "roi": roi_yr,
            "profit": d["pnl"],
        }

    summary = {
        "overall": {
            "n_bets": n,
            "win_rate": wins / n if n > 0 else None,
            "roi": roi,
            "profit": total_pnl,
            "clv_proxy_pp": clv_proxy,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
        },
        "per_year": per_year_out,
        "loss_buckets": {
            "counts": dict(loss_buckets),
            "pct": bucket_pct,
        },
    }
    return bets, summary


def format_summary_table(summary: dict, *, hold: float = DEFAULT_HOLD) -> str:
    """Render headline metrics table for the PR body."""
    lines: list[str] = []
    lines.append(
        f"Walk-forward NBA Phase-2 backtest — market proxy = SRS prior × "
        f"(1 + {hold:.2%} hold)"
    )
    lines.append(
        "  NOTE: No NBA moneyline historical archive available. "
        "CLV proxy is model_prob − srs_prior_prob, NOT true CLV."
    )
    lines.append("")

    o = summary.get("overall", {})
    n = o.get("n_bets", 0)
    wr = o.get("win_rate")
    roi = o.get("roi")
    clv = o.get("clv_proxy_pp")
    profit = o.get("profit", 0.0)
    dd = o.get("max_drawdown")
    sharpe = o.get("sharpe")

    wr_s = f"{wr*100:.1f}%" if wr is not None else "n/a"
    roi_s = f"{roi*100:+.2f}%" if roi is not None else "n/a"
    clv_s = f"{clv*100:+.2f}pp" if clv is not None else "n/a"
    profit_s = f"${profit:+,.0f}" if profit is not None else "n/a"
    dd_s = f"${dd:,.0f}" if dd is not None else "n/a"
    sharpe_s = f"{sharpe:.3f}" if sharpe is not None else "n/a"

    lines.append(f"  n_bets={n}  win_rate={wr_s}  ROI={roi_s}  profit={profit_s}")
    lines.append(f"  clv_proxy={clv_s}  max_drawdown={dd_s}  sharpe={sharpe_s}")
    lines.append("")

    lines.append("  Per-year breakdown:")
    for yr, d in sorted(summary.get("per_year", {}).items()):
        wr_yr = d.get("win_rate")
        roi_yr = d.get("roi")
        wr_str = f"{wr_yr*100:.1f}%" if wr_yr is not None else "n/a"
        roi_str = f"{roi_yr*100:+.2f}%" if roi_yr is not None else "n/a"
        lines.append(
            f"    {yr}: n={d['n_bets']} win%={wr_str} ROI={roi_str} profit=${d['profit']:+,.0f}"
        )

    lines.append("")
    lines.append("  Loss bucket breakdown:")
    buckets = summary.get("loss_buckets", {})
    counts = buckets.get("counts", {})
    pcts = buckets.get("pct", {})
    for bucket, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {bucket}: {cnt} ({pcts.get(bucket, 0):.1f}%)")

    return "\n".join(lines)
