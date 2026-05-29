"""Favorite-longshot bias flag.

If the market's implied probability on the favorite is materially higher than
the blended model's probability for that favorite, the favorite is overpriced.
That's the "chalk-overpriced" flag.
"""

from __future__ import annotations

from collections import defaultdict

from ..config import CHALK_OVERPRICED_DELTA
from ..types import Event, Side, american_to_prob, devig_two_way


def _avg_implied(event: Event) -> dict[Side, float]:
    sums: dict[Side, list[float]] = defaultdict(list)
    for ln in event.lines:
        sums[ln.side].append(american_to_prob(ln.american))
    out: dict[Side, float] = {}
    if sums.get("home"):
        out["home"] = sum(sums["home"]) / len(sums["home"])
    if sums.get("away"):
        out["away"] = sum(sums["away"]) / len(sums["away"])
    return out


def detect(event: Event, delta: float = CHALK_OVERPRICED_DELTA) -> str | None:
    """Return a signal label if chalk is overpriced vs the model, else None."""
    if event.blended_home_prob is None:
        return None
    avg = _avg_implied(event)
    if "home" not in avg or "away" not in avg:
        return None
    home_imp, away_imp = devig_two_way(avg["home"], avg["away"])
    blended_home = event.blended_home_prob
    blended_away = 1.0 - blended_home
    # Identify favorite by market.
    if home_imp >= away_imp:
        fav_imp, fav_model = home_imp, blended_home
    else:
        fav_imp, fav_model = away_imp, blended_away
    if fav_imp > fav_model + delta:
        return "chalk-overpriced"
    return None
