"""Adaptive reweighting: per-sport softmax over a hybrid Brier+ROI score.

Hybrid mode (default, ``brier_roi_hybrid``):

    score_i  = (NAIVE_BRIER - brier_i) + λ · clip(roi_i, -0.2, +0.2)
    weight_i = softmax(β · score_i)

where ``NAIVE_BRIER = 0.25`` (a fair coin) and ``β``, ``λ`` are
configurable. Sources with ``brier_i > NAIVE_BRIER`` for the sport are
EXCLUDED outright — they're worse than guessing, and we don't want them in
the pool.

Per-sport minimum-sample gates:
    nfl → 30   (short seasons)
    nba/mlb → 50
    atp/wta → 50

Output schema (v2):

    {
      "schema": "v2",
      "global": { source: weight, ... },
      "by_sport": { sport: { source: weight, ... }, ... },
      "excluded": { sport: [source, ...], ... },   // why we dropped them
      "mode": str,
      "min_events": { sport: int, ... },
      "beta": float, "lambda": float
    }

The blender prefers per-sport weights when the event's sport is known and
falls back to the global pool otherwise.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from ..config import (
    SOURCE_SCOREBOARD_PATH,
    hybrid_beta,
    hybrid_lambda,
)
from ..db import insert_weight_snapshot
from .blend import save_weights

log = logging.getLogger(__name__)

VALID_MODES = ("brier", "log_loss", "roi", "brier_roi_hybrid")
NAIVE_BRIER = 0.25  # a 50/50 coin scores Brier 0.25
ROI_CLIP = 0.20  # clip rolling ROI to ±20% to limit single-source dominance

PER_SPORT_MIN_EVENTS: dict[str, int] = {
    "nfl": 30,
    "nba": 50,
    "mlb": 50,
    "atp": 50,
    "wta": 50,
}


def weight_mode() -> str:
    val = os.getenv("ACCURACY_WEIGHT_MODE", "brier_roi_hybrid").strip().lower()
    if val not in VALID_MODES:
        log.warning("ACCURACY_WEIGHT_MODE=%s not recognised; falling back to brier", val)
        return "brier"
    return val


def min_events_for(sport: str | None) -> int:
    if sport and sport in PER_SPORT_MIN_EVENTS:
        return PER_SPORT_MIN_EVENTS[sport]
    try:
        return int(os.getenv("FLASHCAT_MIN_EVENTS_FOR_WEIGHT", "50"))
    except Exception:
        return 50


def _softmax(values: dict[str, float], beta: float) -> dict[str, float]:
    if not values:
        return {}
    scaled = {k: beta * v for k, v in values.items()}
    m = max(scaled.values())
    exps = {k: math.exp(v - m) for k, v in scaled.items()}
    total = sum(exps.values())
    if total <= 0:
        n = len(values)
        return {k: 1.0 / n for k in values}
    return {k: v / total for k, v in exps.items()}


def softmax(values: dict[str, float], temperature: float = 4.0) -> dict[str, float]:
    """Backward-compatible alias for ``_softmax``.

    Older callers (and the unit tests) used ``temperature`` to mean the
    softmax sharpness — i.e. the β multiplier on each score.
    """
    return _softmax(values, temperature)


def _row_signal(row: dict, mode: str) -> float | None:
    brier = row.get("brier")
    roi = row.get("roi")
    log_loss = row.get("log_loss")
    if mode == "brier":
        return -float(brier) if brier is not None else None
    if mode == "log_loss":
        return -float(log_loss) if log_loss is not None else None
    if mode == "roi":
        return float(roi) if roi is not None else None
    return None  # hybrid handled separately


def _hybrid_score(row: dict, lam: float) -> float | None:
    """Hybrid score: Brier improvement vs naive + λ · clipped ROI."""
    brier = row.get("brier")
    if brier is None:
        return None
    score = NAIVE_BRIER - float(brier)
    roi = row.get("roi")
    if roi is not None:
        try:
            roi_clipped = max(-ROI_CLIP, min(ROI_CLIP, float(roi)))
            score += lam * roi_clipped
        except Exception:
            pass
    return score


def _compute_pool_weights(
    rows: dict[str, dict],
    *,
    mode: str,
    beta: float,
    lam: float,
    min_n: int,
) -> tuple[dict[str, float], list[dict]]:
    """Filter low-sample / worse-than-naive rows and softmax over score.

    Returns ``(weights, excluded)`` where excluded is a list of
    ``{source, reason, brier, roi, n_events}`` dicts for the scoreboard.
    """
    eligible: dict[str, dict] = {}
    excluded: list[dict] = []
    for k, v in rows.items():
        if k == "flashcat-blended":
            continue
        if not isinstance(v, dict):
            continue
        n = int(v.get("n_events") or 0)
        if n < min_n:
            excluded.append({
                "source": k, "reason": f"n<{min_n}",
                "brier": v.get("brier"), "roi": v.get("roi"), "n_events": n,
            })
            continue
        brier = v.get("brier")
        if brier is not None and float(brier) > NAIVE_BRIER:
            excluded.append({
                "source": k, "reason": f"brier>{NAIVE_BRIER}",
                "brier": brier, "roi": v.get("roi"), "n_events": n,
            })
            continue
        if brier is None and v.get("roi") is None and v.get("log_loss") is None:
            excluded.append({
                "source": k, "reason": "no_signal",
                "brier": None, "roi": None, "n_events": n,
            })
            continue
        eligible[k] = v
    if not eligible:
        return {}, excluded
    if mode == "brier_roi_hybrid":
        scores: dict[str, float] = {}
        for k, v in eligible.items():
            s = _hybrid_score(v, lam)
            if s is not None:
                scores[k] = s
        return _softmax(scores, beta), excluded
    signals: dict[str, float] = {}
    for k, v in eligible.items():
        s = _row_signal(v, mode)
        if s is not None:
            signals[k] = s
    return (_softmax(signals, beta) if signals else {}), excluded


def update_weights(
    scoreboard_path: Path | None = None,
    *,
    beta: float | None = None,
    lam: float | None = None,
    mode: str | None = None,
) -> dict:
    """Read the scoreboard, compute per-sport + global weights, persist them."""
    sp = scoreboard_path or SOURCE_SCOREBOARD_PATH
    if not sp.exists():
        return {}
    with open(sp) as f:
        scoreboard = json.load(f)
    if not isinstance(scoreboard, dict):
        return {}

    mode = mode or weight_mode()
    beta = beta if beta is not None else hybrid_beta()
    lam = lam if lam is not None else hybrid_lambda()

    sources = scoreboard.get("sources", {}) or {}
    global_weights, global_excluded = _compute_pool_weights(
        sources, mode=mode, beta=beta, lam=lam, min_n=min_events_for(None),
    )

    per_sport_rows: dict[str, dict] = {}
    excluded_by_sport: dict[str, list] = {"global": global_excluded}
    min_events_by_sport: dict[str, int] = {"global": min_events_for(None)}
    for sport, p in (scoreboard.get("per_sport") or {}).items():
        srcs = (p or {}).get("sources") or {}
        min_n = min_events_for(sport)
        weights, excluded = _compute_pool_weights(
            srcs, mode=mode, beta=beta, lam=lam, min_n=min_n,
        )
        if weights:
            per_sport_rows[sport] = weights
        excluded_by_sport[sport] = excluded
        min_events_by_sport[sport] = min_n

    payload = {
        "schema": "v2",
        "mode": mode,
        "beta": beta,
        "lambda": lam,
        "min_events": min_events_by_sport,
        "global": global_weights,
        "by_sport": per_sport_rows,
        "excluded": excluded_by_sport,
    }
    save_weights(payload)
    insert_weight_snapshot(global_weights, datetime.now(timezone.utc).isoformat())
    return payload
