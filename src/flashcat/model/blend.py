"""Weighted-average probability blender."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..config import SOURCE_WEIGHTS_PATH
from ..types import Event
from .calibration import calibrate_sport, load_coefficients
from .pick import pick_side


def load_weights(path: Path | None = None) -> dict:
    """Load weights as a v2 payload.

    Returns a dict shaped like::

        {"schema": "v2",
         "global": {source: weight, ...},
         "by_sport": {sport: {source: weight, ...}, ...}}

    If the file on disk is the legacy v1 flat ``{source: weight}`` mapping,
    we promote it into the v2 shape with no per-sport breakdown.
    """
    p = path or SOURCE_WEIGHTS_PATH
    if not p.exists():
        return {"schema": "v2", "global": {}, "by_sport": {}}
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"schema": "v2", "global": {}, "by_sport": {}}
    if data.get("schema") == "v2":
        data.setdefault("global", {})
        data.setdefault("by_sport", {})
        return data
    # Legacy v1 (flat mapping).
    return {"schema": "v2", "global": dict(data), "by_sport": {}}


def save_weights(weights: dict, path: Path | None = None) -> None:
    p = path or SOURCE_WEIGHTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(weights, f, indent=2, sort_keys=True)


def weights_for_sport(weights: dict, sport: str | None) -> dict[str, float]:
    """Resolve the source-weight dict for an event of the given sport.

    Prefers ``by_sport[sport]`` if non-empty; falls back to ``global``; falls
    back to an empty dict (which the blender treats as uniform weighting).
    """
    if not isinstance(weights, dict):
        return {}
    by_sport = weights.get("by_sport") if weights.get("schema") == "v2" else None
    if isinstance(by_sport, dict) and sport in by_sport and by_sport[sport]:
        return dict(by_sport[sport])
    if weights.get("schema") == "v2":
        return dict(weights.get("global") or {})
    # Legacy flat shape.
    return {k: v for k, v in weights.items() if isinstance(v, (int, float))}


def _resolve_weights(sources: Iterable[str], weights: dict[str, float]) -> dict[str, float]:
    """Equal-weight any source not yet in the weights file."""
    out: dict[str, float] = {}
    for s in sources:
        out[s] = float(weights.get(s, 1.0))
    total = sum(out.values())
    if total <= 0:
        # All zero — fall back to uniform
        n = max(1, len(out))
        return {k: 1.0 / n for k in out}
    return {k: v / total for k, v in out.items()}


def blend_event(
    event: Event,
    weights: dict | None = None,
    calibration: dict | None = None,
) -> Event:
    """Compute blended home win prob, write pick + pick_prob, return event.

    If ``calibration`` is provided (per-sport Platt coefficients), the
    blended prob is passed through ``σ(α + β · logit(p))`` as a final step.
    """
    weights = weights if weights is not None else load_weights()
    if not event.source_probs:
        event.blended_home_prob = None
        event.pick = None
        event.pick_prob = None
        return event
    src_names = [p.source for p in event.source_probs]
    sport_weights = weights_for_sport(weights, event.sport)
    w_norm = _resolve_weights(src_names, sport_weights)
    blended = sum(p.home_win_prob * w_norm.get(p.source, 0.0) for p in event.source_probs)
    blended = max(0.0, min(1.0, blended))
    if calibration:
        blended = calibrate_sport(blended, event.sport, calibration)
    event.blended_home_prob = blended
    side, side_prob = pick_side(event, blended)
    event.pick = side
    event.pick_prob = side_prob
    return event


def blend_events(
    events: list[Event],
    weights: dict | None = None,
    calibration: dict | None = None,
) -> list[Event]:
    weights = weights if weights is not None else load_weights()
    if calibration is None:
        calibration = load_coefficients()
    return [blend_event(e, weights, calibration) for e in events]
