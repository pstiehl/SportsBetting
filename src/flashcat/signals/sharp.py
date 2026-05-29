"""Sharp-money / reverse-line-movement + cross-book dispersion."""

from __future__ import annotations

from collections import defaultdict

from ..config import BOOK_DISPERSION_THRESHOLD
from ..types import Event, american_to_prob


def _avg_implied_by_side(event: Event, *, opening: bool) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for ln in event.lines:
        if ln.is_opening != opening:
            continue
        sums[ln.side].append(american_to_prob(ln.american))
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def detect_rlm(event: Event) -> str | None:
    """Reverse line movement: opening home implied prob lower than current,
    OR opening away implied prob lower than current — i.e., the line moved
    against the side the public is typically loaded on.

    Phase-1 simplification: with no %-of-bets feed, we flag any meaningful
    move (≥2 percentage points implied prob) and identify which side it moved toward.
    """
    open_avg = _avg_implied_by_side(event, opening=True)
    close_avg = _avg_implied_by_side(event, opening=False)
    if not open_avg or not close_avg:
        return None
    if "home" not in open_avg or "home" not in close_avg:
        return None
    delta_home = close_avg["home"] - open_avg["home"]
    if delta_home >= 0.02:
        return "reverse-line-movement-toward-home"
    if delta_home <= -0.02:
        return "reverse-line-movement-toward-away"
    return None


def detect_dispersion(event: Event, threshold: float = BOOK_DISPERSION_THRESHOLD) -> str | None:
    """Cross-book dispersion on the underdog side > threshold."""
    side_probs: dict[str, list[float]] = defaultdict(list)
    for ln in event.lines:
        if ln.is_opening:
            continue
        side_probs[ln.side].append(american_to_prob(ln.american))
    if not side_probs.get("home") or not side_probs.get("away"):
        return None
    home_avg = sum(side_probs["home"]) / len(side_probs["home"])
    away_avg = sum(side_probs["away"]) / len(side_probs["away"])
    dog_side = "home" if home_avg < away_avg else "away"
    dog_vals = side_probs[dog_side]
    if len(dog_vals) < 2:
        return None
    spread = max(dog_vals) - min(dog_vals)
    if spread > threshold:
        return "book-dispersion-dog"
    return None


def detect(event: Event) -> list[str]:
    """Return all sharp/dispersion signals firing on this event."""
    out: list[str] = []
    rlm = detect_rlm(event)
    if rlm:
        out.append(rlm)
    disp = detect_dispersion(event)
    if disp:
        out.append(disp)
    return out
