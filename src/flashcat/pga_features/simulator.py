"""Flat-$100 simulator for the walk-forward PGA backtest.

A PORT of ``wta_features.simulator``. Phil's spec: place a hypothetical $100
wager on every model-graded matchup. No production edge gate on the headline
numbers (we ALSO emit the +3pp-gated subset for the report). Golf-specific
loss buckets.

Payout: when the matchup closing decimal odds are archived we use them
directly for PnL, so the payout is the true closing price. When they're
absent we reconstruct from the devigged closing matchup prob plus a standard
book hold. CLV proxy is computed off the devigged closing matchup
probability (``market_prob_home``) — it measures how far the model's pick
prob sits above the closing implied prob. Positive CLV proxy != profit; it's
the necessary calibration signal.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from ..nfl_features.model import FoldResult

log = logging.getLogger(__name__)

DEFAULT_STAKE = 100.0
PRODUCTION_EDGE_GATE_PP = 0.03  # +3pp; mirrors MLB/CFB/NFL/ATP/WTA
DEFAULT_HOLD = 0.045            # fallback book hold when no real decimal odds


def _classify_loss(bet: dict, features: dict, ctx: dict) -> str:
    """Bucket a losing PGA matchup bet by most-probable narrative cause.

    Buckets (priority order):

      pure_variance          — model pick prob in [0.45, 0.55]. In golf H2H
                               matchups the outcome is famously coin-flippy
                               (two comparable players over 4 rounds), so
                               this bucket is expected to dominate.
      line_moved_against     — model edge vs market < 1pp (lost to vig)
      favorite_upset         — picked a strong favorite (prob > 0.70) and lost
      course_fit_wrong       — course-tier finish form favored our pick by
                               > 10pp of quality and we still lost
      skill_signal_wrong     — the DataGolf win%/skill delta strongly favored
                               our pick (win_pct_log_ratio > 0.5 in our
                               direction) and we lost
      form_signal_wrong      — recent finish-quality form favored our pick by
                               > 10pp and we lost
      fatigue_disadvantage   — picked the player with materially LESS rest
                               (rest_days_diff against our pick > 7 days)
      h2h_signal_wrong       — prior H2H history favored our pick (> 60% share
                               in our direction) and we lost
      generic                — fallback
    """
    p = bet["pick_prob"]
    edge = bet["edge_pp"]
    pick_home = bet["pick_home"]

    if 0.45 <= p <= 0.55:
        return "pure_variance"
    if abs(edge) < 0.01:
        return "line_moved_against"
    if p > 0.70:
        return "favorite_upset"

    # Course-fit wrong — course_tier_quality_l10_diff signed toward our pick.
    course = features.get("course_tier_quality_l10_diff")
    if course is not None:
        if pick_home and course > 0.10:
            return "course_fit_wrong"
        if (not pick_home) and course < -0.10:
            return "course_fit_wrong"

    # Skill signal wrong — win_pct_log_ratio favored our pick.
    slr = features.get("win_pct_log_ratio")
    if slr is not None:
        if pick_home and slr > 0.5:
            return "skill_signal_wrong"
        if (not pick_home) and slr < -0.5:
            return "skill_signal_wrong"

    # Recent finish form wrong.
    form = features.get("finish_quality_l10_diff")
    if form is not None:
        if pick_home and form > 0.10:
            return "form_signal_wrong"
        if (not pick_home) and form < -0.10:
            return "form_signal_wrong"

    # Fatigue disadvantage — our pick had less rest.
    rest = features.get("rest_days_diff")
    if rest is not None:
        if pick_home and rest < -7:
            return "fatigue_disadvantage"
        if (not pick_home) and rest > 7:
            return "fatigue_disadvantage"

    # H2H signal wrong.
    h2h = features.get("h2h_home_share")
    if h2h is not None:
        if pick_home and h2h > 0.60:
            return "h2h_signal_wrong"
        if (not pick_home) and h2h < 0.40:
            return "h2h_signal_wrong"

    return "generic"


@dataclass
class BetResult:
    match_date: str
    event_id: str
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
    course_tier: str

    def as_row(self) -> dict:
        return {
            "match_date": self.match_date,
            "event_id": self.event_id,
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
            "course_tier": self.course_tier,
        }


def simulate_flat_stake(
    fold_results: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    edge_gate_pp: float | None = None,
) -> tuple[list[BetResult], dict]:
    """Simulate flat $100 bets on every model-graded PGA matchup.

    Returns (bets, summary). When ``edge_gate_pp`` is non-None, only bets
    with ``|edge_pp| >= edge_gate_pp`` are kept. The summary always reports
    ``n_bets``, ``win_rate``, ``roi``, ``clv_proxy_pp``, ``max_drawdown``,
    ``sharpe``, ``profit``, ``loss_buckets``.
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
            pick_home = pred["home_prob"] >= 0.5
            pick_prob = pred["home_prob"] if pick_home else 1.0 - pred["home_prob"]
            mkt_pick = mkt if pick_home else 1.0 - mkt

            home_dec = pred.get("home_decimal")
            away_dec = pred.get("away_decimal")
            picked_dec = home_dec if pick_home else away_dec
            if picked_dec is not None and picked_dec > 1.0:
                decimal = float(picked_dec)
            else:
                implied_with_hold = min(0.999, mkt_pick * (1.0 + DEFAULT_HOLD))
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
            ctx = {"event_label": pred.get("event_label")}
            loss_bucket = None if won else _classify_loss(bet_dict, features, ctx)
            br = BetResult(
                match_date=pred["match_date"],
                event_id=pred["event_id"],
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
                course_tier=pred.get("course_tier", ""),
            )
            bets.append(br)
            running += pnl
            cumulative_pnl.append(running)

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
    # Max drawdown — peak-to-trough on cumulative PnL.
    peak = -float("inf")
    max_dd = 0.0
    for v in cumulative_pnl:
        if v > peak:
            peak = v
        dd = v - peak
        if dd < max_dd:
            max_dd = dd
    summary["max_drawdown"] = max_dd
    # Sharpe (per-bet, annualized to ~2000 PGA H2H matchups/season — ~45
    # events * ~48 matchups after adjacent pairing of a ~144-player field).
    pnls = [b.pnl for b in bets]
    mean = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            summary["sharpe"] = (mean / sd) * math.sqrt(2000)
    # Loss buckets.
    buckets: dict[str, int] = defaultdict(int)
    for b in bets:
        if b.loss_bucket:
            buckets[b.loss_bucket] += 1
    summary["loss_buckets"] = dict(buckets)
    return bets, summary
