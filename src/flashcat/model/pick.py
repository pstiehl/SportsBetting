"""Decision rule: $100 flat on the higher blended-probability side, with
tie-breaker on the underdog when blended prob is within [0.48, 0.52]."""

from __future__ import annotations

from collections import defaultdict

from ..config import TIE_BREAK_BAND
from ..types import Event, Side, american_to_prob


def _favorite_side(event: Event) -> Side | None:
    """Determine favorite side from average implied prob across books."""
    if not event.lines:
        return None
    side_probs: dict[Side, list[float]] = defaultdict(list)
    for line in event.lines:
        side_probs[line.side].append(american_to_prob(line.american))
    if not side_probs.get("home") or not side_probs.get("away"):
        return None
    home_avg = sum(side_probs["home"]) / len(side_probs["home"])
    away_avg = sum(side_probs["away"]) / len(side_probs["away"])
    if home_avg == away_avg:
        return None
    return "home" if home_avg > away_avg else "away"


def pick_side(event: Event, blended_home_prob: float) -> tuple[Side, float]:
    """Pick the side we'd bet, returning (side, blended prob for that side).

    Decision rule:
    - If blended_home_prob is inside the tie-break band (0.48–0.52), bet the underdog
      by moneyline (Phil's "favorite is a sucker's bet" tie-breaker).
    - Otherwise, bet the side with higher blended probability.
    - If we can't determine a favorite for the tie-break, fall back to higher-prob side.
    """
    lo, hi = TIE_BREAK_BAND
    in_band = lo <= blended_home_prob <= hi
    fav = _favorite_side(event)
    if in_band and fav is not None:
        underdog: Side = "away" if fav == "home" else "home"
        prob = blended_home_prob if underdog == "home" else 1.0 - blended_home_prob
        return underdog, prob
    if blended_home_prob >= 0.5:
        return "home", blended_home_prob
    return "away", 1.0 - blended_home_prob
