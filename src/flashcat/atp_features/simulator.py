"""Flat-$100 simulator for the walk-forward ATP backtest.

Phil's spec: place a hypothetical $100 wager on every model-graded match.
No production edge gate on the headline numbers (we ALSO emit the
+3pp-gated subset for the report). ATP-specific loss buckets.

Payout: tennis-data.co.uk archives REAL closing decimal odds (Pinnacle
preferred). We use those directly for PnL when present, so the payout is
the true closing price, not a reconstructed one. CLV proxy is computed
off the devigged closing market probability (``market_prob_home``) — it
measures how far the model's pick prob sits above the closing implied
prob. Positive CLV proxy ≠ profit; it's the necessary calibration signal.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from ..nfl_features.model import FoldResult

log = logging.getLogger(__name__)

DEFAULT_STAKE = 100.0
PRODUCTION_EDGE_GATE_PP = 0.03  # +3pp; mirrors MLB/CFB/NFL
DEFAULT_HOLD = 0.045            # fallback book hold when no real decimal odds


def _classify_loss(bet: dict, features: dict, ctx: dict) -> str:
    """Bucket a losing ATP bet by most-probable narrative cause.

    Buckets (priority order):

      pure_variance          — model pick prob in [0.45, 0.55]
      line_moved_against     — model edge vs market < 1pp (lost to vig)
      favorite_upset         — picked a strong favorite (prob > 0.70) and lost
      surface_form_wrong     — surface-specific form favored our pick by
                               > 10pp and we still lost
      ranking_signal_wrong   — ranking-points log-ratio strongly favored our
                               pick (> 0.5 in log-odds terms) and we lost
      fatigue_disadvantage   — picked the player with materially LESS rest
                               (rest_days_diff against our pick > 3 days)
      h2h_signal_wrong       — H2H history favored our pick (> 60% share in
                               our direction) and we lost
      best_of_5_variance     — Grand Slam best-of-5 loss with pick in the
                               [0.55, 0.70] band (long-format variance)
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

    # Surface form wrong — surface_win_pct_l20_diff signed toward our pick.
    surf = features.get("surface_win_pct_l20_diff")
    if surf is not None:
        if pick_home and surf > 0.10:
            return "surface_form_wrong"
        if (not pick_home) and surf < -0.10:
            return "surface_form_wrong"

    # Ranking signal wrong — rank_points_log_ratio favored our pick.
    rlr = features.get("rank_points_log_ratio")
    if rlr is not None:
        if pick_home and rlr > 0.5:
            return "ranking_signal_wrong"
        if (not pick_home) and rlr < -0.5:
            return "ranking_signal_wrong"

    # Fatigue disadvantage — our pick had less rest.
    rest = features.get("rest_days_diff")
    if rest is not None:
        if pick_home and rest < -3:
            return "fatigue_disadvantage"
        if (not pick_home) and rest > 3:
            return "fatigue_disadvantage"

    # H2H signal wrong.
    h2h = features.get("h2h_home_share")
    if h2h is not None:
        if pick_home and h2h > 0.60:
            return "h2h_signal_wrong"
        if (not pick_home) and h2h < 0.40:
            return "h2h_signal_wrong"

    # Best-of-5 long-format variance.
    if features.get("best_of_5", 0.0) > 0.5 and 0.55 <= p <= 0.70:
        return "best_of_5_variance"

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
    surface: str

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
            "surface": self.surface,
        }


def simulate_flat_stake(
    fold_results: list[FoldResult],
    *,
    stake: float = DEFAULT_STAKE,
    edge_gate_pp: float | None = None,
) -> tuple[list[BetResult], dict]:
    """Simulate flat $100 bets on every model-graded ATP match.

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

            # Real closing decimal odds for the picked side, when archived.
            home_dec = pred.get("home_decimal")
            away_dec = pred.get("away_decimal")
            picked_dec = home_dec if pick_home else away_dec
            if picked_dec is not None and picked_dec > 1.0:
                decimal = float(picked_dec)
            else:
                # Fallback: reconstruct from devigged prob + standard hold.
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
            ctx = {"series": pred.get("series"), "round": pred.get("round")}
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
                surface=pred.get("surface", ""),
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
    # Sharpe (per-bet, annualized to ~2600 ATP main-tour matches/season).
    pnls = [b.pnl for b in bets]
    mean = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        sd = math.sqrt(var)
        if sd > 0:
            summary["sharpe"] = (mean / sd) * math.sqrt(2600)
    # Loss buckets.
    buckets: dict[str, int] = defaultdict(int)
    for b in bets:
        if b.loss_bucket:
            buckets[b.loss_bucket] += 1
    summary["loss_buckets"] = dict(buckets)
    return bets, summary
