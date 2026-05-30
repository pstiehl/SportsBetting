"""Decision rule: pick the side where the blended model beats the devigged
market — i.e. the side whose blended probability is higher than the market's
devigged implied probability on that side.

Falls back to "higher blended probability" when the market is unavailable, so
the function still works for events with no live odds (e.g. some research
slates).

A tie-break preserves Phil's "favorite is a sucker's bet" rule near the
market: when blended_home_prob is inside ``TIE_BREAK_BAND`` (default
0.48–0.52), bet the underdog by moneyline. This only fires when the model
doesn't disagree meaningfully with the market.
"""

from __future__ import annotations

from collections import defaultdict

from ..config import TIE_BREAK_BAND
from ..types import Event, Side, american_to_prob, devig_two_way


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


def _market_devigged_home_prob(event: Event) -> float | None:
    """Average book-implied probs, devigged. ``None`` if no two-sided market."""
    if not event.lines:
        return None
    by_side: dict[Side, list[float]] = defaultdict(list)
    for ln in event.lines:
        by_side[ln.side].append(american_to_prob(ln.american))
    if not by_side.get("home") or not by_side.get("away"):
        return None
    h = sum(by_side["home"]) / len(by_side["home"])
    a = sum(by_side["away"]) / len(by_side["away"])
    h_d, _ = devig_two_way(h, a)
    return h_d


def pick_side(event: Event, blended_home_prob: float) -> tuple[Side, float]:
    """Pick the side we'd bet, returning (side, blended prob for that side).

    Decision rule:
    - If blended_home_prob is inside the tie-break band (0.48–0.52), bet the
      underdog by moneyline (Phil's tie-breaker). This is a near-zero-edge
      band; the underdog rule reflects the favorite-longshot bias.
    - Else if a two-sided devigged market is available, bet the side where
      ``blended_side_prob > market_devigged_side_prob`` — that's the side
      with positive edge by construction. (If they're exactly equal, fall
      through to the higher-blended-prob rule.)
    - Else, fall back to the side with higher blended probability.

    The returned ``pick_prob`` is the blended probability of the picked side
    (not the market's). Downstream staking uses this together with the
    devigged market to compute the signed edge.
    """
    lo, hi = TIE_BREAK_BAND
    in_band = lo <= blended_home_prob <= hi
    fav = _favorite_side(event)
    if in_band and fav is not None:
        underdog: Side = "away" if fav == "home" else "home"
        prob = blended_home_prob if underdog == "home" else 1.0 - blended_home_prob
        return underdog, prob

    market_home = _market_devigged_home_prob(event)
    if market_home is not None:
        market_away = 1.0 - market_home
        blended_away = 1.0 - blended_home_prob
        # Pick the side where blended beats devigged market. Edge is
        # symmetric (home_edge = -away_edge), so picking the side with the
        # larger (blended - market) on it guarantees the picked side has
        # non-negative edge.
        if blended_home_prob - market_home > blended_away - market_away:
            return "home", blended_home_prob
        if blended_away - market_away > blended_home_prob - market_home:
            return "away", blended_away
        # Tie: fall through to higher-blended-prob rule.

    if blended_home_prob >= 0.5:
        return "home", blended_home_prob
    return "away", 1.0 - blended_home_prob
