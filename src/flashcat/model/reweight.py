"""Adaptive reweighting: per-sport softmax over (negative Brier) and/or ROI.

Modes (ACCURACY_WEIGHT_MODE env):
  - ``brier`` (default until calibrated): softmax over -Brier per sport
  - ``log_loss``: softmax over -log-loss per sport
  - ``roi``: softmax over ROI per sport
  - ``brier_roi_hybrid``: 0.5 * softmax(-Brier) + 0.5 * softmax(ROI) per sport

Sources with ``n_events < MIN_EVENTS`` (default 50) are excluded from the
weight pool. They still appear in the scoreboard but don't influence the blend.

The output schema (v2) is:

    {
      "schema": "v2",
      "global": { source: weight, ... },           // legacy/back-compat
      "by_sport": { sport: { source: weight, ... }, ... }
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

from ..config import SOURCE_SCOREBOARD_PATH
from ..db import insert_weight_snapshot
from .blend import save_weights

log = logging.getLogger(__name__)

VALID_MODES = ("brier", "log_loss", "roi", "brier_roi_hybrid")


def weight_mode() -> str:
    val = os.getenv("ACCURACY_WEIGHT_MODE", "brier_roi_hybrid").strip().lower()
    if val not in VALID_MODES:
        log.warning("ACCURACY_WEIGHT_MODE=%s not recognised; falling back to brier", val)
        return "brier"
    return val


def min_events() -> int:
    try:
        return int(os.getenv("FLASHCAT_MIN_EVENTS_FOR_WEIGHT", "50"))
    except Exception:
        return 50


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


def _row_signal(row: dict, mode: str) -> float | None:
    """Pick the (higher-is-better) signal value to softmax over for one source row."""
    brier = row.get("brier")
    roi = row.get("roi")
    log_loss = row.get("log_loss")
    if mode == "brier":
        return -float(brier) if brier is not None else None
    if mode == "log_loss":
        return -float(log_loss) if log_loss is not None else None
    if mode == "roi":
        return float(roi) if roi is not None else None
    # hybrid handled by caller
    return None


def _blend_hybrid(rows: dict[str, dict], temperature: float) -> dict[str, float]:
    """Hybrid: 0.5 * softmax(-Brier) + 0.5 * softmax(ROI). Sources missing
    ROI fall back to the Brier softmax alone."""
    brier_vals = {k: -float(v["brier"]) for k, v in rows.items() if v.get("brier") is not None}
    roi_vals = {k: float(v["roi"]) for k, v in rows.items() if v.get("roi") is not None}
    if not roi_vals:
        return softmax(brier_vals, temperature=temperature)
    if not brier_vals:
        return softmax(roi_vals, temperature=temperature)
    s_brier = softmax(brier_vals, temperature=temperature)
    s_roi = softmax(roi_vals, temperature=temperature)
    out: dict[str, float] = {}
    for k in set(s_brier) | set(s_roi):
        out[k] = 0.5 * s_brier.get(k, 0.0) + 0.5 * s_roi.get(k, 0.0)
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def _compute_pool_weights(
    rows: dict[str, dict],
    *,
    mode: str,
    temperature: float,
    min_n: int,
) -> dict[str, float]:
    """Filter low-sample rows and softmax over the configured signal."""
    eligible: dict[str, dict] = {}
    for k, v in rows.items():
        if k == "flashcat-blended":
            continue
        if not isinstance(v, dict):
            continue
        n = int(v.get("n_events") or 0)
        if n < min_n:
            continue
        if v.get("brier") is None and v.get("roi") is None and v.get("log_loss") is None:
            continue
        eligible[k] = v
    if not eligible:
        return {}
    if mode == "brier_roi_hybrid":
        return _blend_hybrid(eligible, temperature=temperature)
    signals: dict[str, float] = {}
    for k, v in eligible.items():
        s = _row_signal(v, mode)
        if s is not None:
            signals[k] = s
    return softmax(signals, temperature=temperature) if signals else {}


def update_weights(
    scoreboard_path: Path | None = None,
    temperature: float = 4.0,
    min_n: int | None = None,
    mode: str | None = None,
) -> dict:
    """Read the scoreboard, compute per-sport + global weights, persist them.

    Returns the full v2 payload that was written.
    """
    sp = scoreboard_path or SOURCE_SCOREBOARD_PATH
    if not sp.exists():
        return {}
    with open(sp) as f:
        scoreboard = json.load(f)
    if not isinstance(scoreboard, dict):
        return {}

    mode = mode or weight_mode()
    min_n = min_n if min_n is not None else min_events()

    # Global weights come from the flat per-source scoreboard (which is keyed
    # by ``<sport>:<source>`` in multi-sport runs and bare source name in
    # single-sport runs).
    sources = scoreboard.get("sources", {}) or {}
    global_weights = _compute_pool_weights(
        sources, mode=mode, temperature=temperature, min_n=min_n,
    )

    # Per-sport pools.
    per_sport_rows: dict[str, dict] = {}
    for sport, p in (scoreboard.get("per_sport") or {}).items():
        srcs = (p or {}).get("sources") or {}
        weights = _compute_pool_weights(
            srcs, mode=mode, temperature=temperature, min_n=min_n,
        )
        if weights:
            per_sport_rows[sport] = weights

    payload = {
        "schema": "v2",
        "mode": mode,
        "min_events": min_n,
        "temperature": temperature,
        "global": global_weights,
        "by_sport": per_sport_rows,
    }
    save_weights(payload)
    insert_weight_snapshot(global_weights, datetime.now(timezone.utc).isoformat())
    return payload
