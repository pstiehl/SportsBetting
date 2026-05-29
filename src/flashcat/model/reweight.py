"""Adaptive reweighting: softmax over (negative Brier score) by source."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..config import SOURCE_SCOREBOARD_PATH
from ..db import insert_weight_snapshot
from .blend import save_weights


def softmax(values: dict[str, float], temperature: float = 4.0) -> dict[str, float]:
    """Numerically stable softmax. Higher temperature = sharper concentration."""
    if not values:
        return {}
    scaled = {k: temperature * v for k, v in values.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    total = sum(exps.values())
    if total <= 0:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in exps.items()}


def update_weights(
    scoreboard_path: Path | None = None,
    temperature: float = 4.0,
    min_events: int = 20,
) -> dict[str, float]:
    """Read the source scoreboard, compute new weights, persist them."""
    sp = scoreboard_path or SOURCE_SCOREBOARD_PATH
    if not sp.exists():
        return {}
    with open(sp) as f:
        scoreboard = json.load(f)
    # New scoreboard format wraps sources under a 'sources' key.
    sources = scoreboard.get("sources", scoreboard) if isinstance(scoreboard, dict) else {}
    # Negative Brier → higher is better. We softmax that.
    neg_brier: dict[str, float] = {}
    for source, row in sources.items():
        if source == "flashcat-blended":
            continue  # don't reweight the model itself into its own inputs
        if not isinstance(row, dict):
            continue
        n = int(row.get("n_events", 0))
        b = row.get("brier")
        if b is None or n < min_events:
            continue
        neg_brier[source] = -float(b)
    if not neg_brier:
        return {}
    new_weights = softmax(neg_brier, temperature=temperature)
    save_weights(new_weights)
    insert_weight_snapshot(new_weights, datetime.now(timezone.utc).isoformat())
    return new_weights
