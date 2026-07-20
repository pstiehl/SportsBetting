"""Walk-forward logistic regression for the expanded PGA feature set.

A PORT of ``wta_features.model``. Same structure and hyperparameters as the
tennis / NFL harnesses — L2 logistic (C=1.0) with StandardScaler,
walk-forward 365d train / 30d eval / 30d slide, 120d warmup, leakage gate
asserted in-loop.

We reuse the sport-agnostic ``WalkForwardSplit`` / ``FoldResult`` /
``make_splits`` machinery from ``nfl_features.model`` (the harness the
playbook calls "sport-agnostic") and only re-implement the evaluate loop
because the PGA feature builder takes ``(matchup, rolling, h2h)`` instead of
``(game, rolling, bye_status)``.
"""

from __future__ import annotations

import logging
import math

import numpy as np

# Reuse the sport-agnostic split/fold machinery.
from ..nfl_features.model import (  # noqa: F401
    WalkForwardSplit,
    FoldResult,
    make_splits,
    _logistic,
    _train_one_split,
)
from .feature_builder import (
    FEATURE_NAMES,
    MatchupRow,
    build_features,
    feature_vector,
)

log = logging.getLogger(__name__)


def walk_forward_evaluate(
    matchups: list[MatchupRow],
    rolling: dict,
    h2h: dict,
    *,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
    fill_value: float = 0.0,
) -> list[FoldResult]:
    if not matchups:
        return []
    start = min(m.match_date for m in matchups)
    end = max(m.match_date for m in matchups)
    splits = make_splits(
        start, end,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )

    feats_by_matchup: dict[int, dict] = {}
    for m in matchups:
        f = build_features(m, rolling, h2h)
        if f is None:
            continue
        feats_by_matchup[id(m)] = f
    log.info(
        "walk-forward PGA: %d matchups -> %d with full features; %d splits over %s..%s",
        len(matchups), len(feats_by_matchup), len(splits), start, end,
    )

    out: list[FoldResult] = []
    for split in splits:
        train_examples: list[tuple[list[float], int]] = []
        eval_matchups: list[MatchupRow] = []
        for m in matchups:
            if id(m) not in feats_by_matchup or m.home_won is None:
                continue
            f = feats_by_matchup[id(m)]
            v = feature_vector(f, fill_value=fill_value)
            if split.train_start <= m.match_date <= split.train_end:
                train_examples.append((v, int(m.home_won)))
            elif split.eval_start <= m.match_date <= split.eval_end:
                eval_matchups.append(m)
        fit = _train_one_split(train_examples)
        if fit is None:
            log.debug("skip split %s (only %d train examples)", split, len(train_examples))
            continue
        coef, intercept, scaler = fit
        preds: list[dict] = []
        for m in eval_matchups:
            assert split.eval_start <= m.match_date <= split.eval_end, (
                f"leakage: eval matchup {m.match_date} outside "
                f"[{split.eval_start},{split.eval_end}]"
            )
            f = feats_by_matchup[id(m)]
            v = np.array([feature_vector(f, fill_value=fill_value)], dtype=float)
            vs = scaler.transform(v)
            z = float(coef @ vs[0] + intercept)
            p = _logistic(z)
            preds.append({
                "match_date": m.match_date.isoformat(),
                "event_id": m.event_id,
                "home": m.home,
                "away": m.away,
                "home_prob": p,
                "market_prob_home": m.market_prob_home,
                "skill_bt_prob_home": m.skill_bt_prob_home,
                "home_decimal": m.home_decimal,
                "away_decimal": m.away_decimal,
                "home_won": int(m.home_won),
                "features": f,
                "course_tier": m.course_tier,
                "event_label": m.event_label,
                "season": m.season,
            })
        if preds:
            log_loss = -sum(
                pr["home_won"] * math.log(max(1e-9, pr["home_prob"]))
                + (1 - pr["home_won"]) * math.log(max(1e-9, 1 - pr["home_prob"]))
                for pr in preds
            ) / len(preds)
            acc = sum(1 for pr in preds if (pr["home_prob"] >= 0.5) == bool(pr["home_won"])) / len(preds)
            brier = sum((pr["home_prob"] - pr["home_won"]) ** 2 for pr in preds) / len(preds)
        else:
            log_loss = acc = brier = None
        out.append(FoldResult(
            split=split,
            n_train=len(train_examples),
            n_eval=len(eval_matchups),
            n_picks=len(preds),
            coef=list(coef),
            intercept=intercept,
            log_loss=log_loss,
            accuracy=acc,
            brier=brier,
            predictions=preds,
        ))
    return out
