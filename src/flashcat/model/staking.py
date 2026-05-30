"""Stake sizing — Kelly-fraction and edge-gated $-flat alternatives.

Why this exists:
  Phil's v1 strategy was "$100 flat on every event". That's structurally a
  ``-vig`` strategy whenever the model's edge over the devigged market is
  ≤ the vig itself. The fix is two-pronged:

  1. **Edge gate** — only bet when ``|blended_prob − devig_market_prob|``
     exceeds a configurable threshold (default 3 pp). Coin flips are skipped.
  2. **Kelly fractional staking** — bet a fraction of bankroll proportional
     to edge × (1 / odds). Default 1/4-Kelly to soften variance.

Returns ``None`` (skip) when no live odds are available on the picked side,
when the edge is below threshold, or when the Kelly fraction comes out
non-positive (would mean negative EV).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..config import FLAT_STAKE
from ..types import Event, Side, american_to_decimal, american_to_prob, devig_two_way

DEFAULT_EDGE_THRESHOLD = 0.03  # 3 percentage points
TIE_BREAK_BAND_HALF = 0.02      # 2 pp: "no-bet" rule near the market


@dataclass
class StakeDecision:
    """Result of the staking decision for one event."""

    side: Side
    stake: float
    price: int                  # American odds we'd take
    edge: float                 # blended_prob − devig_market_prob (signed, on picked side)
    reason: str                 # "kelly_quarter" / "flat" / etc.
    skipped_reason: str | None = None  # set when stake == 0


def _book_avg_implied(event: Event, side: Side) -> float | None:
    prices = [
        american_to_prob(ln.american)
        for ln in event.lines
        if ln.side == side and not ln.is_opening
    ]
    if not prices:
        prices = [american_to_prob(ln.american) for ln in event.lines if ln.side == side]
    if not prices:
        return None
    return sum(prices) / len(prices)


def _book_avg_american(event: Event, side: Side) -> int | None:
    prices = [
        ln.american for ln in event.lines if ln.side == side and not ln.is_opening
    ]
    if not prices:
        prices = [ln.american for ln in event.lines if ln.side == side]
    if not prices:
        return None
    return int(round(sum(prices) / len(prices)))


def devigged_prob(event: Event, side: Side) -> float | None:
    """Return the devigged implied probability of ``side`` from book averages."""
    h = _book_avg_implied(event, "home")
    a = _book_avg_implied(event, "away")
    if h is None or a is None:
        return None
    h_d, a_d = devig_two_way(h, a)
    return h_d if side == "home" else a_d


def decide_stake(
    event: Event,
    side: Side,
    blended_side_prob: float,
    *,
    mode: str = "kelly_quarter",
    bankroll: float = 10_000.0,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    no_bet_band: float = TIE_BREAK_BAND_HALF,
) -> StakeDecision:
    """Compute a stake for one event.

    Args:
      event: Event with at least one ``BookLine`` on the picked side.
      side: ``"home"`` or ``"away"`` — chosen by the blender.
      blended_side_prob: Blended model probability of ``side`` winning.
      mode: One of ``flat``, ``kelly_quarter``, ``kelly_half``, ``kelly_full``.
      bankroll: Notional bankroll for Kelly sizing (default $10k).
      edge_threshold: Minimum |blended − market| to take a bet (default 3 pp).
      no_bet_band: If blended ≈ market within ±this band on the picked side,
        skip. Default ±2 pp. Set to 0 to disable.

    Returns a ``StakeDecision``. If ``stake`` is 0, ``skipped_reason`` says why.
    """
    price = _book_avg_american(event, side)
    if price is None:
        return StakeDecision(
            side=side, stake=0.0, price=0, edge=0.0,
            reason=mode, skipped_reason="no_market_price",
        )
    market_prob = devigged_prob(event, side)
    if market_prob is None:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=0.0,
            reason=mode, skipped_reason="no_market_devig",
        )
    edge = blended_side_prob - market_prob

    if abs(edge) < no_bet_band:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="within_no_bet_band",
        )
    if edge < edge_threshold:
        # Negative or insufficient edge on the picked side. Don't bet.
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="edge_below_threshold",
        )

    if mode == "flat":
        return StakeDecision(
            side=side, stake=FLAT_STAKE, price=price, edge=edge, reason="flat",
        )

    # Kelly fractional: f* = (b*p - q) / b where b = decimal odds - 1
    b = american_to_decimal(price) - 1.0
    if b <= 0:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="degenerate_odds",
        )
    p = blended_side_prob
    q = 1.0 - p
    f_full = (b * p - q) / b
    if f_full <= 0:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="kelly_non_positive",
        )
    fraction = {
        "kelly_full": 1.0,
        "kelly_half": 0.5,
        "kelly_quarter": 0.25,
    }.get(mode, 0.25)
    # Variance floor: if fractional Kelly < 0.5% of bankroll, skip the bet.
    f_scaled = f_full * fraction
    if f_scaled < 0.005:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="kelly_below_variance_floor",
        )
    f = max(0.0, min(0.02, f_scaled))  # cap at 2% of bankroll per bet
    stake = round(bankroll * f / 5.0) * 5.0  # round to nearest $5
    if stake <= 0:
        return StakeDecision(
            side=side, stake=0.0, price=price, edge=edge,
            reason=mode, skipped_reason="kelly_below_variance_floor",
        )
    return StakeDecision(
        side=side, stake=stake, price=price, edge=edge, reason=mode,
    )
