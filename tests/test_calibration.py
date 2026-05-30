"""Platt calibration unit tests."""

from __future__ import annotations

import math
import random

from flashcat.model.calibration import (
    apply_platt,
    calibrate_sport,
    fit_platt,
    save_coefficients,
    load_coefficients,
)


def test_fit_recovers_identity_on_calibrated_data():
    """If outcomes already match p, fit should return ~α=0, β=1."""
    rng = random.Random(42)
    rows: list[tuple[float, bool]] = []
    for _ in range(500):
        p = rng.uniform(0.1, 0.9)
        y = rng.random() < p
        rows.append((p, y))
    fit = fit_platt(rows)
    assert fit is not None
    alpha, beta = fit
    assert abs(alpha) < 0.5
    assert 0.7 < beta < 1.4


def test_fit_corrects_overconfidence():
    """If true outcome rate is closer to 0.5 than predicted, β < 1."""
    rng = random.Random(7)
    rows: list[tuple[float, bool]] = []
    for _ in range(500):
        p_pred = rng.uniform(0.1, 0.9)
        # actual prob: regress toward 0.5
        p_real = 0.5 + 0.5 * (p_pred - 0.5)
        y = rng.random() < p_real
        rows.append((p_pred, y))
    fit = fit_platt(rows)
    assert fit is not None
    _, beta = fit
    assert beta < 1.0


def test_fit_returns_none_below_threshold():
    rows = [(0.5, True)] * 10
    assert fit_platt(rows) is None


def test_apply_platt_identity():
    assert abs(apply_platt(0.7, 0.0, 1.0) - 0.7) < 1e-6


def test_persist_and_load(tmp_path, monkeypatch):
    import flashcat.model.calibration as cal
    p = tmp_path / "calibration.json"
    monkeypatch.setattr(cal, "CALIBRATION_PATH", p)
    save_coefficients({"mlb": {"alpha": 0.1, "beta": 0.9, "n": 100}})
    loaded = load_coefficients()
    assert "mlb" in loaded
    assert calibrate_sport(0.6, "mlb", loaded) != 0.6
    # Unknown sport → pass-through
    assert calibrate_sport(0.6, "nfl", loaded) == 0.6
