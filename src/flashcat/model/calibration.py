"""Platt scaling for the blended probability.

After we blend N sources we may still be systematically over- or under-
confident. We fit a logistic regression of outcome on logit(p_blend) on the
rolling 365-day window and apply ``σ(a + b · logit(p_blend))`` as a final
transform.

Coefficients are persisted to ``data/calibration.json`` so the live build
can apply them without re-fitting.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ..config import CALIBRATION_PATH


def _logit(p: float) -> float:
    p = max(1e-3, min(1 - 1e-3, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_platt(rows: list[tuple[float, bool]]) -> tuple[float, float] | None:
    """Fit ``y ~ σ(α + β · logit(p))`` via Newton steps.

    rows: list of (predicted_prob, actual_outcome).
    Returns (alpha, beta) or None if fit doesn't converge / not enough data.
    """
    if len(rows) < 50:
        return None
    xs = [_logit(p) for p, _ in rows]
    ys = [1.0 if y else 0.0 for _, y in rows]
    alpha = 0.0
    beta = 1.0
    for _ in range(60):
        ga = 0.0
        gb = 0.0
        h_aa = 0.0
        h_ab = 0.0
        h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = _sigmoid(alpha + beta * x)
            err = mu - y
            ga += err
            gb += err * x
            w = mu * (1 - mu)
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            return None
        d_alpha = (h_bb * ga - h_ab * gb) / det
        d_beta = (-h_ab * ga + h_aa * gb) / det
        alpha -= d_alpha
        beta -= d_beta
        if abs(d_alpha) + abs(d_beta) < 1e-7:
            break
    if not math.isfinite(alpha) or not math.isfinite(beta):
        return None
    # Sanity guards: keep slope in [0.2, 3.0] — anything outside means the
    # fit either inverted or saturated, both of which mean "don't apply".
    if not (0.2 <= beta <= 3.0):
        return None
    return alpha, beta


def apply_platt(p: float, alpha: float, beta: float) -> float:
    return max(0.001, min(0.999, _sigmoid(alpha + beta * _logit(p))))


def save_coefficients(per_sport: dict, path: Path | None = None) -> None:
    """Persist {sport: {"alpha": ..., "beta": ..., "n": ...}, ...}."""
    p = path or CALIBRATION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "v1",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "per_sport": per_sport,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_coefficients(path: Path | None = None) -> dict:
    p = path or CALIBRATION_PATH
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data.get("per_sport") or {}


def calibrate_sport(p: float, sport: str, coefficients: dict) -> float:
    """Apply per-sport Platt if coefficients exist, else pass-through."""
    entry = coefficients.get(sport)
    if not entry:
        return p
    alpha = entry.get("alpha")
    beta = entry.get("beta")
    if alpha is None or beta is None:
        return p
    return apply_platt(p, float(alpha), float(beta))
