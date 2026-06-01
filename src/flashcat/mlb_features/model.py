"""Walk-forward logistic regression for the expanded MLB feature set.

We use sklearn ``LogisticRegression`` with L2 regularization. Logistic was
chosen over GBM for three reasons:

1. Phil's mental model — coefficients are inspectable per-feature so
   the loss post-mortem can attribute a losing pick to "model overweighted
   the pitcher rating diff" or similar.
2. Calibration — logistic outputs are natively well-calibrated for
   binary classification with reasonable feature counts (vs GBM which
   needs Platt/isotonic post-fit).
3. Sample size — 4 seasons of MLB = ~10K games. GBM with 17 features
   would overfit; logistic is the right complexity floor.

Walk-forward methodology:

* Train on a rolling 365-day window of games strictly BEFORE the
  evaluation date.
* Evaluate the next 30 days out-of-sample.
* Slide forward by 30 days, retrain.

The bucket structure means every evaluation game has been scored by a
model that never saw it (or anything after it). The leakage gate is
asserted on every train/eval split.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .feature_builder import FEATURE_NAMES, GameRow, build_features, feature_vector

log = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """One walk-forward fold: train window + eval window."""

    train_start: date
    train_end: date  # inclusive
    eval_start: date  # exclusive of train_end
    eval_end: date  # inclusive

    def __post_init__(self) -> None:
        assert self.train_end < self.eval_start, (
            f"leakage: train_end {self.train_end} >= eval_start {self.eval_start}"
        )


@dataclass
class FoldResult:
    split: WalkForwardSplit
    n_train: int
    n_eval: int
    n_picks: int  # eval games where model produced a prediction
    coef: list[float]  # logistic coefficients in FEATURE_NAMES order
    intercept: float
    log_loss: float | None
    accuracy: float | None
    brier: float | None
    predictions: list[dict] = field(default_factory=list)  # one per eval pick


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _train_one_split(
    train_examples: list[tuple[list[float], int]],
) -> tuple[np.ndarray, float, StandardScaler] | None:
    """Fit logistic regression on (X, y). Returns (coef, intercept, scaler) or None."""
    if len(train_examples) < 50:
        return None
    X = np.array([e[0] for e in train_examples], dtype=float)
    y = np.array([e[1] for e in train_examples], dtype=int)
    # If all outcomes are the same class, sklearn fails. Bail to caller.
    if len(set(y)) < 2:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(Xs, y)
    return clf.coef_[0], float(clf.intercept_[0]), scaler


def make_splits(
    start: date,
    end: date,
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
) -> list[WalkForwardSplit]:
    """Generate sliding walk-forward splits.

    The first eval window begins ``warmup_days`` after ``start`` (so the
    first fold has at least that much training data even when the rolling
    feature builder needs ~20 games per team to spin up).
    """
    splits: list[WalkForwardSplit] = []
    eval_start = start + timedelta(days=warmup_days)
    while eval_start <= end:
        eval_end = min(eval_start + timedelta(days=eval_window_days - 1), end)
        train_end = eval_start - timedelta(days=1)
        train_start = max(start, train_end - timedelta(days=train_window_days - 1))
        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
            )
        )
        eval_start = eval_end + timedelta(days=1)
    return splits


def walk_forward_evaluate(
    games: list[GameRow],
    snapshots: dict,
    pitcher_rest: dict,
    park_run_env: dict,
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
    fill_value: float = 0.0,
) -> list[FoldResult]:
    """Run the full walk-forward evaluation.

    Returns a list of ``FoldResult``s, each containing per-pick predictions
    so downstream simulators can settle bets and compute ROI / CLV.

    Strict gate: every training example has ``game_date <= split.train_end``
    and every eval example has ``split.eval_start <= game_date <= split.eval_end``.
    Asserted in-loop.
    """
    if not games:
        return []
    start = min(g.game_date for g in games)
    end = max(g.game_date for g in games)
    splits = make_splits(
        start, end,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )

    # Pre-build feature vectors once. Skip games that fail required-feature
    # gate. ``feats_by_game`` keys on id(game).
    feats_by_game: dict[int, dict] = {}
    for g in games:
        f = build_features(g, snapshots, pitcher_rest, park_run_env)
        if f is None:
            continue
        feats_by_game[id(g)] = f

    log.info(
        "walk-forward: %d games -> %d with full features; %d splits over %s..%s",
        len(games), len(feats_by_game), len(splits), start, end,
    )

    out: list[FoldResult] = []
    for split in splits:
        train_examples: list[tuple[list[float], int]] = []
        eval_games: list[GameRow] = []
        for g in games:
            if id(g) not in feats_by_game:
                continue
            if g.home_won is None:
                continue
            f = feats_by_game[id(g)]
            v = feature_vector(f, fill_value=fill_value)
            if split.train_start <= g.game_date <= split.train_end:
                train_examples.append((v, int(g.home_won)))
            elif split.eval_start <= g.game_date <= split.eval_end:
                eval_games.append(g)

        fit = _train_one_split(train_examples)
        if fit is None:
            log.debug("skip split %s (only %d train examples)", split, len(train_examples))
            continue
        coef, intercept, scaler = fit

        preds: list[dict] = []
        for g in eval_games:
            assert g.game_date >= split.eval_start and g.game_date <= split.eval_end, (
                f"leakage: eval game {g.game_date} outside [{split.eval_start},{split.eval_end}]"
            )
            f = feats_by_game[id(g)]
            v = np.array([feature_vector(f, fill_value=fill_value)], dtype=float)
            vs = scaler.transform(v)
            z = float(coef @ vs[0] + intercept)
            p = _logistic(z)
            preds.append({
                "game_date": g.game_date.isoformat(),
                "home": g.home,
                "away": g.away,
                "home_prob": p,
                "elo_prob_home": g.elo_prob_home,
                "rating_prob_home": g.rating_prob_home,
                "home_won": int(g.home_won),
                "features": f,
            })

        # Eval metrics on this fold.
        if preds:
            log_loss = -sum(
                pr["home_won"] * math.log(max(1e-9, pr["home_prob"]))
                + (1 - pr["home_won"]) * math.log(max(1e-9, 1 - pr["home_prob"]))
                for pr in preds
            ) / len(preds)
            acc = sum(
                1 for pr in preds if (pr["home_prob"] >= 0.5) == bool(pr["home_won"])
            ) / len(preds)
            brier = sum(
                (pr["home_prob"] - pr["home_won"]) ** 2 for pr in preds
            ) / len(preds)
        else:
            log_loss = acc = brier = None

        out.append(FoldResult(
            split=split,
            n_train=len(train_examples),
            n_eval=len(eval_games),
            n_picks=len(preds),
            coef=list(coef),
            intercept=intercept,
            log_loss=log_loss,
            accuracy=acc,
            brier=brier,
            predictions=preds,
        ))

    return out
