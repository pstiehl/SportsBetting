#!/usr/bin/env python3
"""Walk-forward CFB Phase-1+2 backtest driver.

Data source: ESPN public scoreboard cached in
  ``data/cache/cfb_schedule_<season>.json``
  (populated by CFBCfbfastREPA._load_schedule() on first run).
  Seasons attempted: 2022, 2023, 2024.
  ~900 FBS regular-season games/season → ~2700 total.

Outputs:
  * data/cfb_walk_forward_backtest.json — machine-readable summary
  * stdout: headline table + loss buckets

This script is OPT-IN. It is NOT invoked by ``flashcat all`` and CI does
not depend on it. The production per-sport gate stays UNTOUCHED.

Usage::

    PYTHONPATH=src python scripts/cfb_walk_forward_backtest.py
    PYTHONPATH=src python scripts/cfb_walk_forward_backtest.py \\
        --start 2022-08-01 --end 2024-12-31

Note on data availability:
    The CFBD API (collegefootballdata.com) requires a paid/signup API key
    that we don't have. The ESPN public scoreboard fallback provides game
    results but NOT PPA/EPA data. The market proxy therefore uses a
    points-scored-based synthetic PPA derived from ESPN scores themselves.
    This is a weaker proxy than real EPA — documented as HARNESS_ONLY if
    n_bets < 200.
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

from flashcat.config import DATA_DIR  # noqa: E402
from flashcat.cfb_features import (  # noqa: E402
    FEATURE_NAMES,
    load_cfb_game_logs,
    fit_rolling_snapshots,
)
from flashcat.cfb_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.cfb_features.simulator import (  # noqa: E402
    DEFAULT_HOLD,
    format_summary_table,
    simulate,
)

log = logging.getLogger("cfb_walk_forward_backtest")


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
    output_json: Path | None = None,
) -> dict:
    """Main backtest driver. Returns summary dict."""

    # 1. Load game logs
    # Determine which seasons to load based on start/end
    seasons = list(range(start.year, end.year + 1))
    # Always include the year before start for warmup data
    if start.year > 2020:
        seasons = list(range(max(2020, start.year - 1), end.year + 1))

    games = load_cfb_game_logs(seasons=seasons)
    if not games:
        log.error(
            "No CFB game logs found — cache is empty. "
            "The ESPN fallback will be attempted on first import. "
            "Run with PYTHONPATH=src to trigger auto-backfill."
        )
        return {
            "error": "no_game_logs",
            "data_blocker": (
                "CFB schedule cache is empty. ESPN fallback runs on import. "
                "Ensure network access and re-run."
            ),
        }

    # Filter to requested window (keep warmup buffer before start)
    from datetime import timedelta
    warmup_date = start - timedelta(days=warmup_days + 60)
    games_buffered = [g for g in games if g.game_date >= warmup_date and g.game_date <= end]
    games_in_window = [g for g in games_buffered if g.game_date >= start]
    log.info(
        "Window %s..%s: %d games in window, %d with warmup buffer",
        start, end, len(games_in_window), len(games_buffered),
    )

    if not games_in_window:
        log.error("No games in evaluation window %s..%s", start, end)
        return {
            "error": "no_games_in_window",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

    # 2. Build rolling snapshots (leakage-safe)
    log.info("Building rolling team snapshots...")
    snapshots = fit_rolling_snapshots(games_buffered)
    log.info("Built %d snapshots", len(snapshots))

    # 3. Walk-forward evaluation
    log.info("Running walk-forward evaluation...")
    folds = walk_forward_evaluate(
        games_buffered,
        snapshots,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )
    log.info("Walk-forward: %d folds completed", len(folds))

    if not folds:
        log.warning("No folds produced — data may be insufficient for the warmup window")

    # 4. Simulate flat-stake bets
    bets, summary = simulate(folds, hold=hold)
    log.info(
        "Simulation: n_bets=%d ROI=%s",
        summary["overall"]["n_bets"],
        f"{summary['overall']['roi']*100:+.2f}%" if summary["overall"]["roi"] is not None else "n/a",
    )

    # 5. Attach window metadata
    summary["window"] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "train_window_days": train_window_days,
        "eval_window_days": eval_window_days,
        "warmup_days": warmup_days,
        "hold": hold,
        "n_seasons": len(set(g.season for g in games_in_window)),
        "seasons": sorted(set(g.season for g in games_in_window)),
    }
    summary["n_games_loaded"] = len(games)
    summary["n_games_in_window"] = len(games_in_window)
    summary["n_games_with_features"] = sum(
        len(f.predictions) for f in folds
    )
    summary["feature_names"] = FEATURE_NAMES
    summary["with_production_edge_gate"] = False  # backtest bypasses gate

    # HARNESS_ONLY flag
    n_bets = summary["overall"]["n_bets"]
    if n_bets < 200:
        summary["verdict"] = (
            "HARNESS_ONLY — insufficient sample (n_bets < 200). "
            "CFB data available via ESPN ESPN fallback but FBS game count per "
            "30-day eval window is sparse (~60-120 games/month in season). "
            "Real closing line data not available; CLV proxy is weak. "
            "Model harness is operational — add more seasons or real odds data."
        )
    else:
        roi = summary["overall"]["roi"]
        clv = summary["overall"]["clv_proxy_pp"]
        if roi is not None and roi > 0.0:
            summary["verdict"] = f"PROFITABLE — ROI {roi*100:+.2f}% over {n_bets} bets (proxy CLV {clv*100:+.2f}pp). Verify with real closing lines."
        elif roi is not None and roi > -0.02:
            summary["verdict"] = f"COINFLIP — ROI {roi*100:+.2f}% (within vig noise). Need real CLV data to conclude."
        else:
            summary["verdict"] = f"LOSING — ROI {roi*100:+.2f}% over {n_bets} bets. Feature set loses to vig; next iteration needed."

    # 6. Print to stdout
    print(format_summary_table(summary, hold=hold))

    # 7. Persist JSON
    if output_json is None:
        output_json = DATA_DIR / "cfb_walk_forward_backtest.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Wrote %s", output_json)

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="CFB walk-forward backtest")
    parser.add_argument("--start", default="2022-08-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--hold", type=float, default=DEFAULT_HOLD)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        train_window_days=args.train_days,
        eval_window_days=args.eval_days,
        warmup_days=args.warmup_days,
        hold=args.hold,
        output_json=Path(args.output) if args.output else None,
    )

    error = result.get("error")
    if error:
        log.error("Backtest failed: %s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
