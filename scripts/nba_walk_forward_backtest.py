#!/usr/bin/env python3
"""Walk-forward NBA Phase-2 backtest driver.

Data source: cached game logs from
  ``data/cache/nba_historical/gamelogs/games_<season>.json``
  (produced by ``scripts/backfill_nba_historical.py`` via nba_api).
  Three seasons available: 2021-22, 2022-23, 2023-24.

Outputs:
  * data/nba_walk_forward_backtest.json — machine-readable summary
  * stdout: headline table + loss buckets

This script is OPT-IN. It is NOT invoked by ``flashcat all`` and CI does
not depend on it. The production per-sport gate stays UNTOUCHED.

Usage::

    PYTHONPATH=src python scripts/nba_walk_forward_backtest.py
    PYTHONPATH=src python scripts/nba_walk_forward_backtest.py \\
        --start 2022-01-01 --end 2024-06-30
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
    load_nba_game_logs,
    fit_rolling_snapshots,
    load_srs_from_db,
)
from flashcat.nba_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.nba_features.simulator import (  # noqa: E402
    DEFAULT_HOLD,
    format_summary_table,
    simulate,
)
from flashcat.source_history import upsert_predictions, upsert_meta  # noqa: E402

log = logging.getLogger("nba_walk_forward_backtest")


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
) -> dict:
    """Main backtest driver. Returns summary dict."""

    # 1. Load game logs (all available seasons; filter by date below)
    games = load_nba_game_logs()
    if not games:
        log.error("No NBA game logs found — check data/cache/nba_historical/gamelogs/")
        return {"error": "no_game_logs"}

    # Filter to requested window (keep a warmup buffer before start for rolling features)
    warmup_date = start - __import__("datetime").timedelta(days=warmup_days + 30)
    games_in_window = [g for g in games if g.game_date >= warmup_date and g.game_date <= end]
    games_for_eval = [g for g in games_in_window if g.game_date >= start]
    log.info(
        "Loaded %d games total; %d in warmup+window (%s→%s); %d in eval window",
        len(games), len(games_in_window), warmup_date, end, len(games_for_eval),
    )

    if not games_in_window:
        log.error("No games in window %s→%s", start, end)
        return {"error": "no_games_in_window"}

    # 2. Build rolling snapshots (leakage-safe; O(n·teams·dates) but fast)
    log.info("Building rolling team snapshots...")
    snapshots = fit_rolling_snapshots(games_in_window)
    log.info("Built %d team-date snapshots", len(snapshots))

    # 3. Load SRS prior from source_history.db
    log.info("Loading SRS prior from source_history.db...")
    srs_lookup = load_srs_from_db(SOURCE_HISTORY_DB_PATH)
    log.info("Loaded %d SRS entries", len(srs_lookup))

    # 4. Walk-forward evaluation
    log.info("Running walk-forward evaluation...")
    folds = walk_forward_evaluate(
        games_in_window, snapshots, srs_lookup,
        train_window_days=train_window_days,
        eval_window_days=eval_window_days,
        warmup_days=warmup_days,
    )
    log.info("Generated %d folds with predictions", len(folds))

    if not folds:
        log.warning(
            "No folds generated — window may be too short (need ~%d warmup days "
            "+ enough games for 50+ training examples per fold)",
            warmup_days,
        )
        return {"error": "no_folds", "warmup_days": warmup_days}

    # 5. Simulate — no edge gate (Phil's spec for honest model eval)
    bets, summary = simulate(folds, hold=hold, edge_gate=None)

    # 6. Also simulate with production +3pp gate (for comparison)
    _, with_gate = simulate(folds, hold=hold, edge_gate=0.03)

    summary["window"] = {"start": start.isoformat(), "end": end.isoformat()}
    summary["n_folds"] = len(folds)
    summary["n_games_loaded"] = len(games_in_window)
    summary["train_window_days"] = train_window_days
    summary["eval_window_days"] = eval_window_days
    summary["hold"] = hold
    summary["features"] = FEATURE_NAMES
    summary["with_production_edge_gate"] = with_gate

    # 7. Ablation: remove rolling FORM features (pt_diff + win_pct) only.
    # B2B/rest are always computable (no rolling history needed) so they
    # remain in the baseline. This isolates the marginal contribution of
    # rolling form vs. schedule context + SRS prior.
    log.info("Running ablation (rolling form features zeroed)...")
    phase2_only_features = {
        "pt_diff_l5_home", "pt_diff_l5_away", "pt_diff_l5_diff",
        "win_pct_l10_home", "win_pct_l10_away", "win_pct_l10_diff",
    }
    import flashcat.nba_features.model as _M
    _orig_bf = _M.build_features

    def _zeroed_bf(g, snaps, srs, **kw):
        feat = _orig_bf(g, snaps, srs, **kw)
        if not feat:
            return feat
        return {k: (None if k in phase2_only_features else v) for k, v in feat.items()}

    _M.build_features = _zeroed_bf
    try:
        base_folds = _M.walk_forward_evaluate(
            games_in_window, snapshots, srs_lookup,
            train_window_days=train_window_days,
            eval_window_days=eval_window_days,
            warmup_days=warmup_days,
        )
    finally:
        _M.build_features = _orig_bf

    _, base_summary = simulate(base_folds, hold=hold, edge_gate=None)
    _, base_gate = simulate(base_folds, hold=hold, edge_gate=0.03)

    summary["phase1_ablation_baseline"] = {
        "description": (
            "Rolling form features (pt_diff_l5, win_pct_l10) zeroed + retrained. "
            "B2B/rest features retained. Same data + same date window — "
            "isolates marginal effect of rolling form vs schedule+SRS prior."
        ),
        "features_zeroed": sorted(phase2_only_features),
        "overall": base_summary["overall"],
        "per_year": base_summary["per_year"],
        "loss_buckets": base_summary["loss_buckets"],
        "with_production_edge_gate": base_gate.get("overall", {}),
    }

    # 8. Persist to source_history.db
    if persist:
        records = []
        for fold in folds:
            for pred in fold.predictions:
                d = pred["game_date"]
                prior_p = pred.get("prior_prob_home")
                clv_pp = (
                    (pred["home_prob"] - prior_p) * 100
                    if prior_p is not None else None
                )
                records.append({
                    "event_id": f"nba-p2:{d}:{pred['home']}:{pred['away']}",
                    "sport": "nba",
                    "source": "nba-flashcat-v2",
                    "commence_time": f"{d}T00:00:00+00:00",
                    "home": pred["home"],
                    "away": pred["away"],
                    "home_prob": pred["home_prob"],
                    "home_won": int(pred["home_won"]),
                    "market_close_home": prior_p,
                    "market_close_decimal": None,
                    "closing_implied_prob": prior_p,
                    "clv_pp": clv_pp,
                })
        n_upserted = upsert_predictions(records, path=SOURCE_HISTORY_DB_PATH)
        log.info("Persisted %d nba-flashcat-v2 predictions", n_upserted)
        o = summary["overall"]
        upsert_meta(
            [{
                "sport": "nba",
                "source": "nba-flashcat-v2",
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "n_events": o["n_bets"],
                "n_bets": o["n_bets"],
                "brier": None,
                "log_loss": None,
                "accuracy": o.get("win_rate"),
                "roi": o.get("roi"),
                "calibration_slope": None,
                "avg_clv_pp": o.get("clv_proxy_pp"),
            }],
            path=SOURCE_HISTORY_DB_PATH,
        )

    # 9. Write JSON output
    if output_json:
        safe = json.loads(json.dumps(summary, default=str))
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(safe, indent=2))
        log.info("Wrote summary to %s", output_json)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-06-30")
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
        print(f"FAILED: {summary['error']}", file=sys.stderr)
        return 2

    print()
    print(format_summary_table(summary, hold=args.hold))

    gate = summary.get("with_production_edge_gate", {})
    gate_o = gate.get("overall", {})
    if gate_o.get("n_bets", 0) > 0:
        roi_s = f"{gate_o['roi']*100:+.2f}%" if gate_o.get("roi") is not None else "n/a"
        print(
            f"\nWith production +3pp edge gate: {gate_o['n_bets']} bets, "
            f"ROI {roi_s}, profit ${gate_o['profit']:+,.0f}"
        )
    else:
        print("\nWith production +3pp edge gate: 0 bets (gate fully suppressed).")

    abl = summary.get("phase1_ablation_baseline", {})
    abl_o = abl.get("overall", {})
    if abl_o.get("n_bets", 0) > 0:
        roi_abl = abl_o.get("roi")
        roi_s = f"{roi_abl*100:+.2f}%" if roi_abl is not None else "n/a"
        print(
            f"\nAblation (SRS+home_court only, Phase-2 features zeroed): "
            f"{abl_o['n_bets']} bets, ROI {roi_s}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
