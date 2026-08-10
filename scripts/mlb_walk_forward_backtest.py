#!/usr/bin/env python3
"""Walk-forward MLB backtest driver — feature-expansion pilot (Phase 1).

Pulls historical MLB games from the 538 archive (committed in
``data/cache/538_mlb_elo.csv``) and optionally augments with Retrosheet
game logs (for 2024+ games where 538 doesn't cover). Computes the
expanded feature set, runs sliding-window logistic regression, and
simulates a $100 flat-stake bet on every model-graded game.

Default window is 2022-01-01 → 2025-12-31. Without external network
access only 2022-01-01 → 2023-10-01 yields graded predictions because:
  * 538 archive ends 2023-10-01
  * Retrosheet supplies 2024+ outcomes but no Elo/pitcher rating prior,
    so the model can't predict those games on the v1 feature set.

This script writes:

  * ``data/source_history.db`` rows under source ``mlb-flashcat-v2``
    (clv_pp populated against the 538 rating_prob proxy).
  * ``data/mlb_walk_forward_backtest.json`` — machine-readable summary.
  * stdout: the headline table + loss post-mortem.

Usage::

    PYTHONPATH=src python scripts/mlb_walk_forward_backtest.py
    PYTHONPATH=src python scripts/mlb_walk_forward_backtest.py \\
        --start 2022-01-01 --end 2025-12-31 --no-retrosheet

This script is OPT-IN. It is not invoked from ``flashcat all`` and CI does
not depend on it running. The production per-sport gate stays UNTOUCHED;
this only adds a new evaluation lens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import DATA_DIR, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.mlb_features import (  # noqa: E402
    GameRow,
    build_features,
    fit_rolling_rates,
    load_538_mlb_games,
    load_retrosheet_games,
)
from flashcat.mlb_features.feature_builder import (  # noqa: E402
    compute_pitcher_rest,
    fit_empirical_park_factor,
    fit_pitcher_form,
    load_park_run_env,
)
from flashcat.mlb_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.mlb_features.simulator import (  # noqa: E402
    DEFAULT_HOLD,
    format_summary_table,
    simulate,
)
from flashcat.source_history import upsert_predictions, upsert_meta  # noqa: E402

log = logging.getLogger("mlb_walk_forward_backtest")


# ---------------------------------------------------------------------------
# Game loading + merging
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _merge_games(
    primary: list[GameRow], secondary: list[GameRow]
) -> list[GameRow]:
    """Merge two lists of GameRow; primary wins, secondary fills holes.

    Match key = (date, sorted team pair). The merge fills in
    pitcher/umpire/park fields onto 538 rows that lack them (since 538
    doesn't ship Retrosheet park_id / umpire_id).
    """
    by_key: dict[tuple, GameRow] = {}
    for g in primary:
        key = (g.game_date, *sorted([g.home, g.away]))
        by_key[key] = g
    for g in secondary:
        key = (g.game_date, *sorted([g.home, g.away]))
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = g
            continue
        # Fill in fields existing row lacks.
        if existing.park_id is None and g.park_id is not None:
            existing.park_id = g.park_id
        if existing.day_night is None and g.day_night is not None:
            existing.day_night = g.day_night
        if existing.home_pitcher_id is None and g.home_pitcher_id is not None:
            existing.home_pitcher_id = g.home_pitcher_id
        if existing.away_pitcher_id is None and g.away_pitcher_id is not None:
            existing.away_pitcher_id = g.away_pitcher_id
        if existing.plate_umpire_id is None and g.plate_umpire_id is not None:
            existing.plate_umpire_id = g.plate_umpire_id
    return list(by_key.values())


def _load_all_games(
    start: date, end: date, *, include_retrosheet: bool
) -> list[GameRow]:
    """Load games for the year range.

    Week-8 change: Retrosheet game logs are now the PRIMARY source. The 538
    archive 404s (shuttered 2023-10-01) and its committed cache is gone, so
    the pitcher/bullpen/park features and the rolling strength prior are all
    derived from Retrosheet box scores. 538 is merged in only if a cache
    happens to be present (fills legacy pitcher-rating columns, otherwise a
    no-op).
    """
    primary: list[GameRow] = []
    if include_retrosheet:
        for yr in range(start.year, end.year + 1):
            rs = load_retrosheet_games(yr)
            rs = [g for g in rs if start <= g.game_date <= end]
            log.info("Loaded %d games from Retrosheet %d", len(rs), yr)
            primary.extend(rs)
    secondary = load_538_mlb_games(start=start, end=end)
    if secondary:
        log.info("Loaded %d games from 538 archive (legacy merge)", len(secondary))
        # 538 fills legacy pitcher-rating columns onto matching Retrosheet rows.
        return _merge_games(primary, secondary)
    if not primary:
        # Last resort: 538-only (no Retrosheet available).
        return secondary
    return primary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    *,
    start: date,
    end: date,
    include_retrosheet: bool = True,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
    hold: float = DEFAULT_HOLD,
    persist: bool = True,
    output_json: Path | None = None,
) -> dict:
    games = _load_all_games(start, end, include_retrosheet=include_retrosheet)
    if not games:
        log.error("No games loaded for %s..%s — aborting", start, end)
        return {"error": "no games"}
    log.info("Total games after merge: %d", len(games))

    # Rolling-rate snapshots (must be computed on the FULL game list because
    # rolling features for game G need games strictly before G).
    snapshots = fit_rolling_rates(games)
    pitcher_rest = compute_pitcher_rest(games)
    parks = load_park_run_env()
    pitcher_form = fit_pitcher_form(games)
    park_factor_emp = fit_empirical_park_factor(games)

    # Walk-forward eval.
    folds = walk_forward_evaluate(
        games, snapshots, pitcher_rest, parks,
        pitcher_form=pitcher_form,
        park_factor_emp=park_factor_emp,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )
    log.info("Generated %d walk-forward folds", len(folds))

    # Simulate at $100 flat. Phil's spec: no edge gate.
    bets, summary = simulate(folds, hold=hold, edge_gate=None)
    summary["window"] = {"start": start.isoformat(), "end": end.isoformat()}
    summary["n_folds"] = len(folds)
    summary["n_games_loaded"] = len(games)
    summary["train_window_days"] = train_window_days
    summary["eval_window_days"] = eval_window_days
    summary["hold"] = hold
    summary["features"] = list(
        # Sourced from feature_builder.FEATURE_NAMES; expose for the PR.
        __import__("flashcat.mlb_features.feature_builder", fromlist=["FEATURE_NAMES"])
        .FEATURE_NAMES
    )

    # Also compute a "with production gate" sub-result so the PR can show
    # what the +3pp gate would have done.
    _, with_gate = simulate(folds, hold=hold, edge_gate=0.03)
    summary["with_production_edge_gate"] = with_gate

    # Honest ablation: re-simulate with the Week-8 pitcher/bullpen/park
    # features force-zeroed (retrain), to isolate their marginal effect.
    # This is the true "baseline vs new" comparison for the PR — both legs
    # use the SAME Retrosheet data + rolling strength prior, so the delta is
    # attributable to the new features alone.
    week8 = {
        "sp_er_l3_diff", "sp_kbb_l5_diff", "sp_hr_l5_diff",
        "bullpen_load_l3_diff", "park_run_env_emp",
    }
    import flashcat.mlb_features.model as _M
    _orig_bf = _M.build_features

    def _zeroed_bf(g, s, r, p, **kw):
        x = _orig_bf(g, s, r, p, **kw)
        if not x:
            return x
        return {k: (None if k in week8 else v) for k, v in x.items()}

    _M.build_features = _zeroed_bf
    try:
        base_folds = _M.walk_forward_evaluate(
            games, snapshots, pitcher_rest, parks,
            pitcher_form=pitcher_form, park_factor_emp=park_factor_emp,
            train_window_days=train_window_days,
            eval_window_days=eval_window_days, warmup_days=warmup_days,
        )
    finally:
        _M.build_features = _orig_bf
    _, base_summary = simulate(base_folds, hold=hold, edge_gate=None)
    _, base_gate = simulate(base_folds, hold=hold, edge_gate=0.03)
    summary["week8_ablation_baseline"] = {
        "overall": base_summary["overall"],
        "per_year": base_summary["per_year"],
        "loss_buckets": base_summary["loss_buckets"],
        "with_production_edge_gate": base_gate.get("overall", {}),
        "note": (
            "Week-8 features (sp_er_l3_diff, sp_kbb_l5_diff, sp_hr_l5_diff, "
            "bullpen_load_l3_diff, park_run_env_emp) force-zeroed + retrained. "
            "Same data, same prior, same bet eligibility — isolates the marginal "
            "effect of the new features."
        ),
    }

    # Persist to source_history.db.
    if persist:
        records = []
        for f in folds:
            for pred in f.predictions:
                d = pred["game_date"]
                rating_p = pred.get("prior_prob_home") or pred.get("rating_prob_home")
                # CLV proxy on the home side (model prob - market prob, in pp).
                clv_pp = (
                    (pred["home_prob"] - rating_p) * 100
                    if rating_p is not None else None
                )
                records.append({
                    "event_id": f"mlb:{d}:{pred['home']}:{pred['away']}",
                    "sport": "mlb",
                    "source": "mlb-flashcat-v2",
                    "commence_time": f"{d}T18:00:00+00:00",
                    "home": pred["home"],
                    "away": pred["away"],
                    "home_prob": pred["home_prob"],
                    "home_won": pred["home_won"],
                    "market_close_home": rating_p,
                    "market_close_decimal": None,
                    "closing_implied_prob": rating_p,
                    "clv_pp": clv_pp,
                })
        n = upsert_predictions(records, path=SOURCE_HISTORY_DB_PATH)
        log.info("Persisted %d mlb-flashcat-v2 predictions to %s", n, SOURCE_HISTORY_DB_PATH)
        # Also write a meta row so the scoreboard sees the new source.
        overall = summary["overall"]
        upsert_meta(
            [{
                "sport": "mlb",
                "source": "mlb-flashcat-v2",
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "n_events": overall["n_bets"],
                "n_bets": overall["n_bets"],
                "brier": None,
                "log_loss": None,
                "accuracy": overall.get("win_rate"),
                "roi": overall.get("roi"),
                "calibration_slope": None,
                "avg_clv_pp": overall.get("clv_proxy_pp"),
            }],
            path=SOURCE_HISTORY_DB_PATH,
        )

    if output_json:
        # Sanitize for JSON
        safe = json.loads(json.dumps(summary, default=str))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(safe, indent=2))
        log.info("Wrote summary to %s", output_json)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--no-retrosheet",
        action="store_true",
        help="Skip Retrosheet augmentation (offline / faster).",
    )
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD)
    parser.add_argument("--no-persist", action="store_true",
                        help="Don't write to source_history.db (useful for dry-run).")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "mlb_walk_forward_backtest.json"),
        help="Where to write the JSON summary (relative or absolute).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    summary = run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        include_retrosheet=not args.no_retrosheet,
        train_window_days=args.train_days,
        eval_window_days=args.eval_days,
        warmup_days=args.warmup_days,
        hold=args.hold,
        persist=not args.no_persist,
        output_json=Path(args.output),
    )
    if "error" in summary:
        print("FAILED:", summary["error"], file=sys.stderr)
        return 2

    print()
    print(format_summary_table(summary, hold=args.hold))

    gate = summary.get("with_production_edge_gate") or {}
    if gate.get("overall", {}).get("n_bets", 0) > 0:
        o = gate["overall"]
        roi_s = f"{o['roi']*100:+.2f}%" if o.get('roi') is not None else "n/a"
        print()
        print(
            f"With production +3pp edge gate applied: "
            f"{o['n_bets']} bets, ROI {roi_s}, profit ${o['profit']:+,.0f}"
        )
    else:
        print()
        print("With production +3pp edge gate applied: 0 bets (gate fully suppressed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
