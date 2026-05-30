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
    blender_min_bets_for_exclusion,
    blender_roi_floor,
    hybrid_beta,
    hybrid_lambda,
)
from ..db import insert_weight_snapshot
from .blend import save_weights

log = logging.getLogger(__name__)

VALID_MODES = ("brier", "log_loss", "roi", "brier_roi_hybrid")
NAIVE_BRIER = 0.25  # a 50/50 coin scores Brier 0.25
ROI_CLIP = 0.20  # clip rolling ROI to ±20% to limit single-source dominance


def _overlay_source_history_for_reweight(per_sport: dict) -> dict:
    """Fold persisted ``source_history.meta`` rows into ``per_sport.sources``.

    Same contract as ``build_site._overlay_source_history_meta`` — scoreboard
    rows win, persisted rows only fill gaps. Kept inline to avoid an import
    cycle with ``flashcat.build_site``.
    """
    try:
        from ..source_history import connect, init_db
    except Exception:  # pragma: no cover
        return per_sport
    try:
        init_db()
        with connect() as c:
            rows = c.execute(
                "SELECT sport, source, n_events, n_bets, brier, log_loss, "
                "accuracy, roi, calibration_slope FROM meta"
            ).fetchall()
    except Exception:  # pragma: no cover
        return per_sport
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["sport"], r["source"])
        cur = best.get(key)
        if cur is None or (r["n_events"] or 0) > (cur["n_events"] or 0):
            best[key] = dict(r)
    out = dict(per_sport)
    for (sport, source), row in best.items():
        if sport not in out:
            continue
        srcs = (out[sport].get("sources") or {})
        if source in srcs:
            continue
        srcs[source] = {
            "n_events": row.get("n_events") or 0,
            "n_bets": row.get("n_bets") or 0,
            "brier": row.get("brier"),
            "log_loss": row.get("log_loss"),
            "accuracy": row.get("accuracy"),
            "roi": row.get("roi"),
            "wins": 0,
            "losses": 0,
            "calibration_slope": row.get("calibration_slope"),
        }
        out[sport] = dict(out[sport])
        out[sport]["sources"] = srcs
    return out

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


def _n_bets(row: dict) -> int:
    """Best-effort n_bets for a source row.

    Prefers an explicit ``n_bets`` field (the post-edge-gate count of graded
    bets), then ``wins + losses`` (scoreboard rows), then ``n_events``
    (graded probabilistic predictions).
    """
    n_bets = row.get("n_bets")
    if n_bets is not None:
        try:
            return int(n_bets)
        except Exception:
            pass
    try:
        wins = int(row.get("wins") or 0)
        losses = int(row.get("losses") or 0)
        if wins + losses > 0:
            return wins + losses
    except Exception:
        pass
    try:
        return int(row.get("n_events") or 0)
    except Exception:
        return 0


def _cap_low_sample_sources(
    weights: dict[str, float],
    rows: dict[str, dict],
    *,
    min_bets_for_exclusion: int,
) -> dict[str, float]:
    """Cap low-sample sources at 1/N of the surviving pool.

    Per spec item (5): a source with n_bets < min_bets_for_exclusion has too
    much noise on its ROI to make an exclude decision, so we keep it in the
    blend but cap its max weight at 1/N (where N is the number of surviving
    sources) so it can't dominate. Mass freed by the cap is redistributed
    proportionally to the uncapped sources.
    """
    if not weights:
        return weights
    n = len(weights)
    if n == 0:
        return weights
    cap = 1.0 / n
    low_sample = {
        k for k in weights
        if _n_bets(rows.get(k) or {}) < min_bets_for_exclusion
    }
    if not low_sample:
        return weights
    capped: dict[str, float] = {}
    freed = 0.0
    for k, w in weights.items():
        if k in low_sample and w > cap:
            freed += (w - cap)
            capped[k] = cap
        else:
            capped[k] = w
    if freed <= 1e-12:
        return capped
    uncapped_total = sum(
        w for k, w in capped.items() if k not in low_sample
    )
    if uncapped_total <= 0:
        # Edge case: everything is low-sample. Renormalize evenly.
        total = sum(capped.values())
        if total <= 0:
            return {k: 1.0 / n for k in capped}
        return {k: w / total for k, w in capped.items()}
    out: dict[str, float] = {}
    for k, w in capped.items():
        if k in low_sample:
            out[k] = w
        else:
            out[k] = w + freed * (w / uncapped_total)
    return out


def _compute_pool_weights(
    rows: dict[str, dict],
    *,
    mode: str,
    beta: float,
    lam: float,
    min_n: int,
    roi_floor: float | None = None,
    min_bets_for_exclusion: int | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Filter low-sample / worse-than-naive rows and softmax over score.

    De-dilution rules (PR "blender de-dilute"):
      * Sources whose ``roi`` is strictly below ``roi_floor`` AND have
        ``n_bets >= min_bets_for_exclusion`` are HARD-EXCLUDED. They get a
        weight of 0 and appear in the excluded list with reason
        ``roi=<v> below floor <floor>``.
      * If applying that rule would leave the pool with fewer than 2
        surviving sources, we instead keep the pool intact (you need at
        least 2 sources to blend meaningfully) and emit a synthetic
        ``min_sources_floor_active`` entry in the excluded list.
      * Low-sample sources (n_bets < min_bets_for_exclusion) keep their
        natural softmax weight but are capped at 1/N afterwards.

    Returns ``(weights, excluded)`` where excluded is a list of
    ``{source, reason, brier, roi, n_events, n_bets}`` dicts for the
    scoreboard.
    """
    if roi_floor is None:
        roi_floor = blender_roi_floor()
    if min_bets_for_exclusion is None:
        min_bets_for_exclusion = blender_min_bets_for_exclusion()

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
                "brier": v.get("brier"), "roi": v.get("roi"),
                "n_events": n, "n_bets": _n_bets(v),
            })
            continue
        brier = v.get("brier")
        if brier is not None and float(brier) > NAIVE_BRIER:
            excluded.append({
                "source": k, "reason": f"brier>{NAIVE_BRIER}",
                "brier": brier, "roi": v.get("roi"),
                "n_events": n, "n_bets": _n_bets(v),
            })
            continue
        if brier is None and v.get("roi") is None and v.get("log_loss") is None:
            excluded.append({
                "source": k, "reason": "no_signal",
                "brier": None, "roi": None,
                "n_events": n, "n_bets": _n_bets(v),
            })
            continue
        eligible[k] = v

    # Hard ROI-floor exclusion (with min-sources fallback).
    roi_failers: list[tuple[str, dict]] = []
    for k, v in eligible.items():
        roi = v.get("roi")
        if roi is None:
            continue
        try:
            roi_f = float(roi)
        except Exception:
            continue
        if _n_bets(v) < min_bets_for_exclusion:
            continue
        if roi_f < roi_floor:
            roi_failers.append((k, v))

    survivors_after_floor = {
        k: v for k, v in eligible.items()
        if k not in {kk for kk, _ in roi_failers}
    }
    if len(survivors_after_floor) >= 2 and roi_failers:
        for k, v in roi_failers:
            excluded.append({
                "source": k,
                "reason": f"roi={float(v.get('roi')):.4f} below floor {roi_floor:.4f}",
                "brier": v.get("brier"), "roi": v.get("roi"),
                "n_events": int(v.get("n_events") or 0),
                "n_bets": _n_bets(v),
            })
        eligible = survivors_after_floor
    elif roi_failers:
        # Would leave <2 survivors — keep everyone, emit fallback marker.
        excluded.append({
            "source": None,
            "reason": "min_sources_floor_active",
            "detail": (
                f"{len(roi_failers)} source(s) below ROI floor {roi_floor:.4f} "
                f"kept in pool to maintain ≥2 sources for blending"
            ),
            "would_exclude": [k for k, _ in roi_failers],
        })

    if not eligible:
        return {}, excluded

    if mode == "brier_roi_hybrid":
        scores: dict[str, float] = {}
        for k, v in eligible.items():
            s = _hybrid_score(v, lam)
            if s is not None:
                scores[k] = s
        weights = _softmax(scores, beta) if scores else {}
    else:
        signals: dict[str, float] = {}
        for k, v in eligible.items():
            s = _row_signal(v, mode)
            if s is not None:
                signals[k] = s
        weights = _softmax(signals, beta) if signals else {}

    # Cap low-sample sources at 1/N of the surviving pool.
    weights = _cap_low_sample_sources(
        weights, eligible,
        min_bets_for_exclusion=min_bets_for_exclusion,
    )
    return weights, excluded


def update_weights(
    scoreboard_path: Path | None = None,
    *,
    beta: float | None = None,
    lam: float | None = None,
    mode: str | None = None,
    roi_floor: float | None = None,
    min_bets_for_exclusion: int | None = None,
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
    roi_floor = roi_floor if roi_floor is not None else blender_roi_floor()
    min_bets_for_exclusion = (
        min_bets_for_exclusion
        if min_bets_for_exclusion is not None
        else blender_min_bets_for_exclusion()
    )

    sources = scoreboard.get("sources", {}) or {}
    global_weights, global_excluded = _compute_pool_weights(
        sources, mode=mode, beta=beta, lam=lam, min_n=min_events_for(None),
        roi_floor=roi_floor, min_bets_for_exclusion=min_bets_for_exclusion,
    )

    # Overlay persistent ``source_history.db.meta`` rows onto the in-memory
    # per-sport sources map. This lets connectors that only publish stats via
    # the backfill scripts (e.g. ``nba-bref-srs-pace`` via
    # ``scripts/backfill_nba_historical.py``) contribute weight to the blender.
    per_sport_overlaid = _overlay_source_history_for_reweight(
        scoreboard.get("per_sport") or {}
    )

    per_sport_rows: dict[str, dict] = {}
    excluded_by_sport: dict[str, list] = {"global": global_excluded}
    min_events_by_sport: dict[str, int] = {"global": min_events_for(None)}
    for sport, p in per_sport_overlaid.items():
        srcs = (p or {}).get("sources") or {}
        min_n = min_events_for(sport)
        weights, excluded = _compute_pool_weights(
            srcs, mode=mode, beta=beta, lam=lam, min_n=min_n,
            roi_floor=roi_floor,
            min_bets_for_exclusion=min_bets_for_exclusion,
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
        "roi_floor": roi_floor,
        "min_bets_for_exclusion": min_bets_for_exclusion,
        "min_events": min_events_by_sport,
        "global": global_weights,
        "by_sport": per_sport_rows,
        "excluded": excluded_by_sport,
    }
    save_weights(payload)
    insert_weight_snapshot(global_weights, datetime.now(timezone.utc).isoformat())
    return payload
