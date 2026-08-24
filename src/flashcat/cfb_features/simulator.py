"""Flat-$100 simulator for the walk-forward CFB backtest — Phase 1+2.

Phil's spec: place a hypothetical $100 wager on EVERY qualifying historical
game during backtest. No production edge gate applied in the base run.

Market proxy:
  No CFB moneyline historical archive is freely available. We use the EPA
  connector's predicted-point-diff → win-probability as the market proxy
  with a 4.5% hold.  CLV is therefore a proxy (model_prob − epa_prior_prob),
  NOT true CLV.

  NOTE: EPA prior uses season-level PPA aggregated across the full season
  (from the ESPN fallback). This is a weak proxy — real line data would
  give better CLV estimates. We document this clearly as HARNESS_ONLY.

CFB Loss buckets:
  upset_heavy_favorite    — model prob > 0.65; big favorite lost (upset)
  turnover_disaster       — margin_volatility was high; proxy for turnover chaos
  line_moved_against      — model edge over proxy < 1pp; lost to vig
  pure_variance           — model prob in [0.45, 0.55]; coinflip
  rolling_signal_wrong    — off/def efficiency strongly backed our pick; we lost
  generic                 — none of the above
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

    # 1. Upset heavy favorite
    if p > 0.65:
        return "upset_heavy_favorite"

    # 2. Coinflip
    if 0.45 <= p <= 0.55:
        return "pure_variance"

    # 3. Lost to vig — edge was < 1pp
    if abs(p - market) < 0.01:
        return "line_moved_against"

    # 4. Margin volatility / turnover chaos — high volatility on our pick team
    vol_home = features.get("margin_volatility_home") or 0.0
    vol_away = features.get("margin_volatility_away") or 0.0
    CFB_VOL_THRESHOLD = 18.0  # std dev of ~18+ pts indicates high chaos
    if pick_home and vol_home > CFB_VOL_THRESHOLD:
        return "turnover_disaster"
    if not pick_home and vol_away > CFB_VOL_THRESHOLD:
        return "turnover_disaster"

    # 5. Rolling efficiency signal was wrong
    net_eff = features.get("net_eff_l5_diff")  # positive = home favored
    if net_eff is not None and abs(net_eff) > 5.0:
        signal_home = net_eff > 0
        if (signal_home and pick_home) or (not signal_home and not pick_home):
            return "rolling_signal_wrong"

    return "generic"


@dataclass
class BetResult:
    game_date: str
    season: Optional[int]
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
            "season": self.season,
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
    per_season: dict[int, dict] = defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0}
    )
    loss_buckets: dict[str, int] = defaultdict(int)

    for fold in folds:
        for pred in fold.predictions:
            home_prob = pred["home_prob"]
            away_prob = 1.0 - home_prob
            home_won = pred["home_won"]

            # Market proxy from EPA prior
            prior = pred.get("prior_prob_home")
            if prior is None:
                # No EPA prior — use flat 0.60 CFB HFA placeholder (stronger than NBA)
                prior = 0.60

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

            season = pred.get("season")
            if season is not None:
                per_season[int(season)]["n"] += 1
                per_season[int(season)]["wins"] += int(won)
                per_season[int(season)]["pnl"] += pnl

            bets.append(
                BetResult(
                    game_date=pred["game_date"],
                    season=pred.get("season"),
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

    # Sharpe (annualized to CFB season; ~700 FBS games/regular season)
    if bets:
        pnl_list = [b.pnl for b in bets]
        mu = sum(pnl_list) / len(pnl_list)
        variance = sum((p - mu) ** 2 for p in pnl_list) / max(len(pnl_list) - 1, 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        ann_factor = math.sqrt(700)
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

    per_season_out = {}
    for s, d in sorted(per_season.items()):
        nn = d["n"]
        per_season_out[str(s)] = {
            "n_bets": nn,
            "win_rate": d["wins"] / nn if nn > 0 else None,
            "roi": d["pnl"] / (nn * stake) if nn > 0 else None,
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
        "per_season": per_season_out,
        "loss_buckets": {
            "counts": dict(loss_buckets),
            "pct": bucket_pct,
        },
        "data_notes": (
            "Market proxy: EPA connector predicted-point-diff → win probability "
            "(season-level PPA from ESPN fallback, no real closing line data). "
            "CLV proxy is model_prob − epa_prior_prob, NOT true CLV. "
            "HARNESS_ONLY if n_bets < 200 — insufficient data for conclusions."
        ),
    }
    return bets, summary


def format_summary_table(summary: dict, *, hold: float = DEFAULT_HOLD) -> str:
    """Render headline metrics table for the PR body."""
    lines: list[str] = []
    lines.append(
        f"Walk-forward CFB Phase-1+2 backtest — market proxy = EPA prior × "
        f"(1 + {hold:.2%} hold)"
    )
    lines.append(
        "  NOTE: No CFB moneyline historical archive available. "
        "CLV proxy is model_prob − epa_prior_prob (season-level PPA), NOT true CLV."
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

    if n < 200:
        lines.append("  ⚠️  HARNESS_ONLY — n_bets < 200, no statistical conclusions.")
        lines.append("")

    lines.append("  Per-season breakdown:")
    for s, d in sorted(summary.get("per_season", {}).items()):
        wr_s2 = d.get("win_rate")
        roi_s2 = d.get("roi")
        wr_str = f"{wr_s2*100:.1f}%" if wr_s2 is not None else "n/a"
        roi_str = f"{roi_s2*100:+.2f}%" if roi_s2 is not None else "n/a"
        lines.append(
            f"    Season {s}: n={d['n_bets']} win%={wr_str} ROI={roi_str} profit=${d['profit']:+,.0f}"
        )

    lines.append("")
    lines.append("  Loss bucket breakdown:")
    buckets = summary.get("loss_buckets", {})
    counts = buckets.get("counts", {})
    pcts = buckets.get("pct", {})
    for bucket, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {bucket}: {cnt} ({pcts.get(bucket, 0):.1f}%)")

    lines.append("")
    notes = summary.get("data_notes")
    if notes:
        lines.append(f"  Data notes: {notes}")

    return "\n".join(lines)
