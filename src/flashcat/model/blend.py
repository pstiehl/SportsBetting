"""Weighted-average probability blender."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ..config import SOURCE_WEIGHTS_PATH
from ..types import Event
from .pick import pick_side


def load_weights(path: Path | None = None) -> dict[str, float]:
    p = path or SOURCE_WEIGHTS_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def save_weights(weights: dict[str, float], path: Path | None = None) -> None:
    p = path or SOURCE_WEIGHTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(weights, f, indent=2, sort_keys=True)


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


def blend_event(event: Event, weights: dict[str, float] | None = None) -> Event:
    """Compute blended home win prob, write pick + pick_prob, return event."""
    weights = weights if weights is not None else load_weights()
    if not event.source_probs:
        event.blended_home_prob = None
        event.pick = None
        event.pick_prob = None
        return event
    src_names = [p.source for p in event.source_probs]
    w_norm = _resolve_weights(src_names, weights)
    blended = sum(p.home_win_prob * w_norm.get(p.source, 0.0) for p in event.source_probs)
    blended = max(0.0, min(1.0, blended))
    event.blended_home_prob = blended
    side, side_prob = pick_side(event, blended)
    event.pick = side
    event.pick_prob = side_prob
    return event


def blend_events(events: list[Event], weights: dict[str, float] | None = None) -> list[Event]:
    weights = weights if weights is not None else load_weights()
    return [blend_event(e, weights) for e in events]
