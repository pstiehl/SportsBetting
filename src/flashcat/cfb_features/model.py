"""Walk-forward logistic regression for the CFB Phase-1+2 feature set.

Methodology (mirrors nba_features/model.py exactly):
* Train on a rolling 365-day window of games strictly BEFORE the eval date.
* Evaluate the next 30 days out-of-sample.
* Slide forward by 30 days, retrain.
* Model: LogisticRegression L2 (C=1.0), StandardScaler normalization.
* Leakage gate: asserted in __post_init__ via split.train_end < split.eval_start.
* Mean imputation computed on train set only; applied to eval set.

CFB-specific notes:
* Regular season spans late Aug → mid-Dec (12-15 game weeks per team).
* Playoffs (CFP) run Dec-Jan.  We include all completed games.
* A 365-day training window spans roughly one full season of history.
* Warmup of 120 days ensures rolling L5 features are populated
  (~5 weeks × ~1 game/week per team = 5 prior games, enough for L5).
* Market proxy: predicted-diff from the EPA connector, converted to
  win probability via the 16.5-sigma normal CDF.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .feature_builder import (
    FEATURE_NAMES,
    CFBGameRow,
    CFBTeamSnapshot,
    build_features,
    feature_vector,
)

log = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """One walk-forward fold."""

    train_start: date
    train_end: date    # inclusive
    eval_start: date   # strictly after train_end
    eval_end: date     # inclusive

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
    log_loss: Optional[float]
    accuracy: Optional[float]
    brier: Optional[float]
    predictions: list[dict] = field(default_factory=list)


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


def _safe_log_loss(probs: list[float], labels: list[int]) -> Optional[float]:
    if not probs:
        return None
    eps = 1e-7
    ll = -sum(
        y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps)
        for p, y in zip(probs, labels)
    )
    return ll / len(probs)


def make_splits(
    start: date,
    end: date,
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
) -> list[WalkForwardSplit]:
    """Generate all non-overlapping evaluation windows."""
    splits: list[WalkForwardSplit] = []
    eval_start = start + timedelta(days=warmup_days)
    while eval_start <= end:
        eval_end = min(eval_start + timedelta(days=eval_window_days - 1), end)
        train_start = eval_start - timedelta(days=train_window_days)
        train_end = eval_start - timedelta(days=1)
        if train_end >= train_start:
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


def _epa_prior_prob(game: CFBGameRow) -> Optional[float]:
    """Market proxy: use the EPA connector to compute a home-win probability.

    This mirrors the 'opening line proxy' concept.  We use the season-level
    PPA differential stored in cache to produce a rough prior.  If not
    available, returns None (model will use 0.55 HFA placeholder).
    """
    try:
        from ..sources.cfb_cfbfastr_epa import (
            CFBCfbfastREPA,
            _load_coeffs,
            _conf_dummy,
            predicted_diff,
            diff_to_home_prob,
        )
        connector = CFBCfbfastREPA(timeout=10.0)
        ppa_data = connector._load_team_ppa([game.season])
        h = ppa_data.get(game.home)
        a = ppa_data.get(game.away)
        if not h or not a:
            return None
        conf_d = _conf_dummy(game.home_conference, game.away_conference)
        coeffs = _load_coeffs(game.season)
        diff = predicted_diff(
            h["off_ppa"], a["off_ppa"],
            h["def_ppa"], a["def_ppa"],
            is_home=True,
            conf_dummy=conf_d,
            coeffs=coeffs,
        )
        return diff_to_home_prob(diff)
    except Exception:
        return None


def walk_forward_evaluate(
    games: list[CFBGameRow],
    snapshots: dict[tuple[str, date], CFBTeamSnapshot],
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
) -> list[FoldResult]:
    """Run the full walk-forward evaluation.

    Returns one FoldResult per eval window.  Folds with < 50 training
    examples are skipped (model can't fit).
    """
    game_dates = sorted({g.game_date for g in games})
    if not game_dates:
        return []

    start = game_dates[0]
    end = game_dates[-1]

    splits = make_splits(
        start, end,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )
    log.info("Generated %d walk-forward splits", len(splits))

    # Pre-build feature dicts for every game (snapshot is already leakage-safe)
    game_feat: dict[str, Optional[dict]] = {}
    for g in games:
        feat = build_features(g, snapshots)
        game_feat[g.game_id] = feat

    # Pre-compute EPA priors (slow — one season-level call, cached)
    # We build a season→connector-loaded PPA map up front
    epa_priors: dict[str, Optional[float]] = {}
    for g in games:
        epa_priors[g.game_id] = _epa_prior_prob(g)

    results: list[FoldResult] = []

    for split in splits:
        # --- Training set ---
        train_games = [
            g for g in games
            if split.train_start <= g.game_date <= split.train_end
            and g.home_won is not None
            and game_feat.get(g.game_id) is not None
        ]

        # Compute training-set feature means for imputation
        feat_sums: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
        feat_counts: dict[str, int] = {name: 0 for name in FEATURE_NAMES}
        for g in train_games:
            feat = game_feat[g.game_id]
            if feat is None:
                continue
            for name in FEATURE_NAMES:
                val = feat.get(name)
                if val is not None:
                    feat_sums[name] += val
                    feat_counts[name] += 1
        fill_mean: dict[str, float] = {
            name: (feat_sums[name] / feat_counts[name]) if feat_counts[name] > 0 else 0.0
            for name in FEATURE_NAMES
        }

        train_examples: list[tuple[list[float], int]] = []
        for g in train_games:
            feat = game_feat[g.game_id]
            if feat is None:
                continue
            vec = feature_vector(feat, fill_mean=fill_mean)
            if vec is None:
                continue
            train_examples.append((vec, int(g.home_won)))

        if len(train_examples) < 50:
            log.debug(
                "Skipping fold %s..%s: only %d training examples",
                split.eval_start, split.eval_end, len(train_examples),
            )
            continue

        # Check for class imbalance
        labels = [e[1] for e in train_examples]
        if len(set(labels)) < 2:
            continue

        X_train = np.array([e[0] for e in train_examples], dtype=float)
        y_train = np.array(labels, dtype=int)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(X_train_s, y_train)

        coef = clf.coef_[0].tolist()
        intercept = float(clf.intercept_[0])

        # --- Eval set ---
        eval_games = [
            g for g in games
            if split.eval_start <= g.game_date <= split.eval_end
            and g.home_won is not None
            and game_feat.get(g.game_id) is not None
        ]

        predictions: list[dict] = []
        eval_probs: list[float] = []
        eval_labels: list[int] = []

        for g in eval_games:
            feat = game_feat[g.game_id]
            if feat is None:
                continue
            vec = feature_vector(feat, fill_mean=fill_mean)
            if vec is None:
                continue
            x_s = scaler.transform([vec])[0]
            logit_val = float(np.dot(coef, x_s) + intercept)
            home_prob = _logistic(logit_val)

            # Market proxy (EPA-based prior)
            prior_prob = epa_priors.get(g.game_id)

            eval_probs.append(home_prob)
            eval_labels.append(int(g.home_won))

            predictions.append({
                "game_id": g.game_id,
                "game_date": g.game_date.isoformat(),
                "season": g.season,
                "week": g.week,
                "home": g.home,
                "away": g.away,
                "home_conference": g.home_conference,
                "away_conference": g.away_conference,
                "home_prob": home_prob,
                "home_won": bool(g.home_won),
                "prior_prob_home": prior_prob,
                "features": {k: feat.get(k) for k in FEATURE_NAMES},
            })

        ll = _safe_log_loss(eval_probs, eval_labels)
        acc: Optional[float] = None
        brier: Optional[float] = None
        if eval_probs:
            preds_binary = [1 if p >= 0.5 else 0 for p in eval_probs]
            acc = sum(p == y for p, y in zip(preds_binary, eval_labels)) / len(eval_labels)
            brier = sum((p - y) ** 2 for p, y in zip(eval_probs, eval_labels)) / len(eval_labels)

        results.append(
            FoldResult(
                split=split,
                n_train=len(train_examples),
                n_eval=len(eval_games),
                n_picks=len(predictions),
                coef=coef,
                intercept=intercept,
                log_loss=ll,
                accuracy=acc,
                brier=brier,
                predictions=predictions,
            )
        )

    log.info("Completed %d folds with predictions", len(results))
    return results
