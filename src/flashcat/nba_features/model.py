"""Walk-forward logistic regression for the expanded NBA feature set.

Mirrors ``flashcat.mlb_features.model`` (same methodology: rolling 365d
train, 30d eval, 30d slide, L2 logistic, StandardScaler) but takes the
NBA-flavored inputs (no pitcher_rest, no park_run_env — NBA features are
all derivable from priors + rolling state).

Why logistic instead of GBM, again: coefficients are inspectable per
feature so the loss post-mortem can attribute a losing pick to
``model overweighted the rest-diff feature`` or similar; logistic is
well-calibrated at this feature count (~18); and the NBA sample size
in source_history.db (~7500 graded games) sits right where GBM would
start overfitting an 18-d space.
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
    n_picks: int
    coef: list[float]
    intercept: float
    log_loss: float | None
    accuracy: float | None
    brier: float | None
    predictions: list[dict] = field(default_factory=list)


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _train_one_split(
    train_examples: list[tuple[list[float], int]],
) -> tuple[np.ndarray, float, StandardScaler] | None:
    if len(train_examples) < 50:
        return None
    X = np.array([e[0] for e in train_examples], dtype=float)
    y = np.array([e[1] for e in train_examples], dtype=int)
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
    rolling_signals: dict,
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
    fill_value: float = 0.0,
) -> list[FoldResult]:
    """Run the full walk-forward NBA evaluation.

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

    feats_by_game: dict[int, dict] = {}
    for g in games:
        f = build_features(g, rolling_signals)
        if f is None:
            continue
        feats_by_game[id(g)] = f

    log.info(
        "NBA walk-forward: %d games -> %d with features; %d splits over %s..%s",
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
                "raptor_prob_home": g.raptor_prob_home,
                "elo_modern_prob_home": g.elo_modern_prob_home,
                "bref_srs_prob_home": g.bref_srs_prob_home,
                "home_won": int(g.home_won),
                "features": f,
            })

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
