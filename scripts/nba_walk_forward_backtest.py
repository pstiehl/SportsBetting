#!/usr/bin/env python3
"""Walk-forward NBA backtest driver — feature-expansion Week 2 (Phase 1).

Pulls historical NBA games from ``data/source_history.db`` (the three
already-backfilled priors: 538-raptor, 538-elo-modern, bref-srs-pace),
joins them on (date, home, away), computes the expanded feature set,
runs sliding-window logistic regression, and simulates a $100 flat-stake
bet on every model-graded game.

Default window is 2022-01-01 → 2024-04-14 (the coverage span of the
backfilled NBA priors). The output JSON is consumed by the source
scoreboard and the weekly Phil report.

Usage::

    PYTHONPATH=src python3 scripts/nba_walk_forward_backtest.py
    PYTHONPATH=src python3 scripts/nba_walk_forward_backtest.py \\
        --start 2022-01-01 --end 2024-04-14 --no-persist

This script is OPT-IN. It is not invoked from ``flashcat all`` and CI
does not depend on it running. The production per-sport gate stays
UNTOUCHED; this only adds a new evaluation lens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import DATA_DIR, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.nba_features import (  # noqa: E402
    FEATURE_NAMES,
    GameRow,
    build_features,
    feature_vector,
    load_nba_games_from_history,
)
from flashcat.nba_features.feature_builder import build_rolling_signals  # noqa: E402
from flashcat.nba_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.nba_features.simulator import (  # noqa: E402
    DEFAULT_HOLD,
    format_summary_table,
    settle_bets,
)
from flashcat.source_history import upsert_predictions, upsert_meta  # noqa: E402

log = logging.getLogger(__name__)


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def run(
    *,
    start: date,
    end: date,
    train_window_days: int = 365,
    eval_window_days: int = 30,
    warmup_days: int = 120,
    hold: float = DEFAULT_HOLD,
    persist: bool = True,
    output_json: Path | None = None,
    db_path: Path | None = None,
) -> dict:
    games = load_nba_games_from_history(
        db_path or SOURCE_HISTORY_DB_PATH, start=start, end=end
    )
    if not games:
        log.error("No NBA games loaded for %s..%s — aborting", start, end)
        return {"error": "no games"}
    log.info("Total NBA games loaded: %d", len(games))

    rolling = build_rolling_signals(games)
    folds = walk_forward_evaluate(
        games, rolling,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )
    log.info("Generated %d walk-forward folds", len(folds))

    bets, summary = settle_bets(folds, hold=hold, edge_gate=None)
    summary["window"] = {"start": start.isoformat(), "end": end.isoformat()}
    summary["n_folds"] = len(folds)
    summary["n_games_loaded"] = len(games)
    summary["train_window_days"] = train_window_days
    summary["eval_window_days"] = eval_window_days
    summary["hold"] = hold
    summary["features"] = list(FEATURE_NAMES)

    # With-production-gate sub-result (+3pp edge requirement).
    _, with_gate = settle_bets(folds, hold=hold, edge_gate=0.03)
    summary["with_production_edge_gate"] = with_gate

    if persist:
        records = []
        for f in folds:
            for pred in f.predictions:
                d = pred["game_date"]
                # Market proxy on home side = consensus of available priors.
                priors_h = [
                    v for v in (
                        pred.get("raptor_prob_home"),
                        pred.get("elo_modern_prob_home"),
                        pred.get("bref_srs_prob_home"),
                    ) if v is not None
                ]
                consensus_h = sum(priors_h) / len(priors_h) if priors_h else None
                clv_pp = (
                    (pred["home_prob"] - consensus_h) * 100
                    if consensus_h is not None else None
                )
                records.append({
                    "event_id": f"nba:{d}:{pred['home']}:{pred['away']}",
                    "sport": "nba",
                    "source": "nba-flashcat-v2",
                    "commence_time": f"{d}T23:00:00+00:00",
                    "home": pred["home"],
                    "away": pred["away"],
                    "home_prob": pred["home_prob"],
                    "home_won": pred["home_won"],
                    "market_close_home": consensus_h,
                    "market_close_decimal": None,
                    "closing_implied_prob": consensus_h,
                    "clv_pp": clv_pp,
                })
        n = upsert_predictions(records, path=SOURCE_HISTORY_DB_PATH)
        log.info("Persisted %d nba-flashcat-v2 predictions to %s", n, SOURCE_HISTORY_DB_PATH)
        overall = summary["overall"]
        upsert_meta(
            [{
                "sport": "nba",
                "source": "nba-flashcat-v2",
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
        safe = json.loads(json.dumps(summary, default=str))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(safe, indent=2))
        log.info("Wrote summary to %s", output_json)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-04-14")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD)
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "nba_walk_forward_backtest.json"),
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
