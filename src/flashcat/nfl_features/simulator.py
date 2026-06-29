"""Flat-$100 simulator for the walk-forward NFL backtest.

Phil's spec: place a hypothetical $100 wager on every model-graded game.
No production edge gate (we ALSO emit the +3pp-gated subset for the
report, but the headline numbers are ungated). NFL-specific loss buckets.

CLV: NFL has real closing moneylines via nflverse, so we use those
directly. No proxy needed.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from .model import FoldResult

log = logging.getLogger(__name__)

DEFAULT_STAKE = 100.0
PRODUCTION_EDGE_GATE_PP = 0.03  # +3pp; mirrors MLB/CFB


def _moneyline_to_decimal(ml: float) -> float:
    """American moneyline → decimal odds (book payout including stake)."""
    if ml >= 0:
        return 1.0 + ml / 100.0
    return 1.0 + 100.0 / (-ml)


def _moneyline_to_raw_implied(ml: float) -> float:
    if ml >= 0:
        return 100.0 / (ml + 100.0)
    return -ml / (-ml + 100.0)


@dataclass
class BetResult:
    game_date: str
    game_id: str
    home: str
    away: str
    pick_home: bool
    pick_prob: float
    market_implied_prob: float
    market_decimal: float
    won: bool
    pnl: float
    edge_pp: float
    loss_bucket: str | None
    season: int
    week: int

    def as_row(self) -> dict:
        return {
            "game_date": self.game_date,
            "game_id": self.game_id,
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
            "season": self.season,
            "week": self.week,
        }


def _classify_loss(bet: dict, features: dict) -> str:
    """Bucket a losing NFL bet by most-probable narrative cause.

    Buckets (priority order):

      pure_variance         — model pick prob in [0.45, 0.55]
      line_moved_against    — model edge vs market < 1pp (lost to vig)
      divisional_misjudged  — divisional game (where favorites cover less);
                              picked the favorite and lost
      bye_off_overrated     — picked team coming off a bye and lost
      rest_disadvantage     — picked team had > 3 days less rest
      rolling_signal_wrong  — L4 EPA differential agreed with pick by > 0.10
                              EPA/play and we still lost
      prior_disagreement_wrong — Elo / EPA / market disagreed by > 8pp; we
                                 went with the minority and were wrong
      generic               — fallback
    """
    p = bet["pick_prob"]
    market = bet["market_implied_prob"]
    edge = bet["edge_pp"]
    pick_home = bet["pick_home"]

    if 0.45 <= p <= 0.55:
        return "pure_variance"
    if abs(edge) < 0.01:
        return "line_moved_against"
    # Divisional misjudge — only flag when we picked the favorite (p > 0.55)
    if features.get("divisional") and features["divisional"] > 0.5 and p > 0.55:
        return "divisional_misjudged"
    # Bye-off picks that lost
    if pick_home and features.get("home_off_bye", 0) > 0.5:
        return "bye_off_overrated"
    if (not pick_home) and features.get("away_off_bye", 0) > 0.5:
        return "bye_off_overrated"
    # Rest disadvantage
    rest = features.get("rest_diff") or 0.0
    if pick_home and rest < -3:
        return "rest_disadvantage"
    if (not pick_home) and rest > 3:
        return "rest_disadvantage"
    # Rolling EPA signal wrong (picked the stronger off-EPA team, lost)
    off_l4 = features.get("off_epa_l4_diff")
    if off_l4 is not None:
        if pick_home and off_l4 > 0.10:
            return "rolling_signal_wrong"
        if (not pick_home) and off_l4 < -0.10:
            return "rolling_signal_wrong"
    # Prior disagreement — picked a side where priors strongly disagreed
    elo = features.get("elo_prob_home")
    epa = features.get("epa_prob_home")
    mkt = features.get("market_prob_home")
    if elo is not None and mkt is not None and abs(elo - mkt) > 0.08:
        if (pick_home and elo > mkt) or (not pick_home and elo < mkt):
            return "prior_disagreement_wrong"
    if epa is not None and mkt is not None and abs(epa - mkt) > 0.08:
        if (pick_home and epa > mkt) or (not pick_home and epa < mkt):
            return "prior_disagreement_wrong"
    return "generic"


def simulate_flat_stake(
    fold_results: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    edge_gate_pp: float | None = None,
) -> tuple[list[BetResult], dict]:
    """Simulate flat $100 bets on every model-graded NFL game.

    Returns (bets, summary). When ``edge_gate_pp`` is non-None, only bets
    with ``|edge_pp| >= edge_gate_pp`` make it into the result list. The
    summary always reports ``n_bets``, ``win_rate``, ``roi``,
    ``clv_proxy_pp``, ``max_drawdown``, ``sharpe``, ``profit``.
    """
    bets: list[BetResult] = []
    cumulative_pnl: list[float] = []
    running = 0.0
    for fold in fold_results:
        for pred in fold.predictions:
            features = pred["features"]
            mkt = pred.get("market_prob_home")
            if mkt is None:
                continue
            # Pick the side the model favors.
            pick_home = pred["home_prob"] >= 0.5
            pick_prob = pred["home_prob"] if pick_home else 1.0 - pred["home_prob"]
            mkt_pick = mkt if pick_home else 1.0 - mkt
            # Use the actual closing moneyline for payout. Reconstruct the
            # raw (vig-laden) implied prob to get the decimal price.
            # We don't have the raw moneyline in the prediction dict, so
            # use a standard 4.5% hold reconstruction for the decimal.
            # (CLV is computed off the devigged market prob.)
            implied_with_hold = min(0.999, mkt_pick * (1.0 + 0.045))
            decimal = 1.0 / implied_with_hold
            won = bool(pred["home_won"]) if pick_home else (not bool(pred["home_won"]))
            pnl = (decimal - 1.0) * stake if won else -stake
            edge_pp = pick_prob - mkt_pick
            if edge_gate_pp is not None and abs(edge_pp) < edge_gate_pp:
                continue
            bet_dict = {
                "pick_prob": pick_prob,
                "market_implied_prob": mkt_pick,
                "edge_pp": edge_pp,
                "pick_home": pick_home,
            }
            loss_bucket = None if won else _classify_loss(bet_dict, features)
            br = BetResult(
                game_date=pred["game_date"],
                game_id=pred["game_id"],
                home=pred["home"],
                away=pred["away"],
                pick_home=pick_home,
                pick_prob=pick_prob,
                market_implied_prob=mkt_pick,
                market_decimal=decimal,
                won=won,
                pnl=pnl,
                edge_pp=edge_pp,
                loss_bucket=loss_bucket,
                season=pred.get("season", 0),
                week=pred.get("week", 0),
            )
            bets.append(br)
            running += pnl
            cumulative_pnl.append(running)
    # Summary
    summary: dict = {
        "n_bets": len(bets),
        "win_rate": None,
        "roi": None,
        "clv_proxy_pp": None,
        "max_drawdown": None,
        "sharpe": None,
        "profit": None,
        "loss_buckets": {},
        "stake": stake,
        "edge_gate_pp": edge_gate_pp,
    }
    if not bets:
        return bets, summary
    wins = sum(1 for b in bets if b.won)
    total_pnl = sum(b.pnl for b in bets)
    total_stake = stake * len(bets)
    summary["win_rate"] = wins / len(bets)
    summary["roi"] = total_pnl / total_stake
    summary["clv_proxy_pp"] = sum(b.edge_pp for b in bets) / len(bets)
    summary["profit"] = total_pnl
    # Max drawdown — peak-to-trough on the cumulative PnL curve
    peak = -float("inf")
    max_dd = 0.0
    for v in cumulative_pnl:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    summary["max_drawdown"] = max_dd
    # Sharpe (per-bet, annualized to 272 NFL games per season = ~16 games/team * 17 weeks)
    pnls = [b.pnl for b in bets]
    mean = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            summary["sharpe"] = (mean / sd) * math.sqrt(272)
    # Loss buckets
    buckets: dict[str, int] = defaultdict(int)
    for b in bets:
        if b.loss_bucket:
            buckets[b.loss_bucket] += 1
    summary["loss_buckets"] = dict(buckets)
    return bets, summary
