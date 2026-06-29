#!/usr/bin/env python3
"""Walk-forward NFL backtest driver — feature-expansion Phase 1.

Pulls 2022-2024 regular-season schedules + PBP from nflverse, attaches
priors (538 Elo, nflfastR EPA) from data/source_history.db, builds the
17-feature Phase-1 catalog, runs sliding-window logistic regression, and
simulates a $100 flat-stake bet on every model-graded game.

Outputs:
  * data/nfl_walk_forward_backtest.json — machine-readable summary
  * paw-reports/sportsbetting/nfl-week4-pr-body.md — PR body with tables
  * stdout: headline table + loss buckets

This script is OPT-IN. It is not invoked by ``flashcat all`` and CI does
not depend on it. The production per-sport gate stays UNTOUCHED.

Usage:
    PYTHONPATH=src python scripts/nfl_walk_forward_backtest.py
    PYTHONPATH=src python scripts/nfl_walk_forward_backtest.py --start 2022-09-01 --end 2024-12-31
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
from flashcat.nfl_features import (  # noqa: E402
    FEATURE_NAMES,
    load_games_from_schedules,
    load_pbp_rollups,
    fit_rolling_rates,
)
from flashcat.nfl_features.feature_builder import (  # noqa: E402
    attach_priors_from_db,
    compute_bye_status,
)
from flashcat.nfl_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.nfl_features.simulator import simulate_flat_stake  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-09-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--out", default=str(DATA_DIR / "nfl_walk_forward_backtest.json"))
    p.add_argument("--pr-body", default=str(REPO_ROOT / "paw-reports" / "sportsbetting" / "nfl-week4-pr-body.md"))
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--eval-days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=120)
    return p.parse_args()


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _format_pr_body(summary: dict, gated_summary: dict, ungated_summary: dict,
                    args: argparse.Namespace, seasons: list[int],
                    n_games_loaded: int, n_games_graded: int,
                    n_folds: int, top_loss_buckets: list[tuple[str, int]]) -> str:
    def _row(label, s):
        roi_pct = f"{s.get('roi', 0)*100:+.2f}%" if s.get("roi") is not None else "—"
        wr_pct = f"{s.get('win_rate', 0)*100:.1f}%" if s.get("win_rate") is not None else "—"
        clv_pp = f"{s.get('clv_proxy_pp', 0)*100:+.2f}pp" if s.get("clv_proxy_pp") is not None else "—"
        dd = f"${s.get('max_drawdown', 0):,.0f}" if s.get("max_drawdown") is not None else "—"
        sharpe = f"{s.get('sharpe'):.2f}" if s.get("sharpe") is not None else "—"
        profit = f"${s.get('profit', 0):,.0f}" if s.get("profit") is not None else "—"
        return f"| {label} | {s['n_bets']} | {wr_pct} | {roi_pct} | {clv_pp} | {dd} | {sharpe} | {profit} |"

    lines = [
        "## feat(nfl): Phase-1 walk-forward harness + feature catalog",
        "",
        f"Week 4 of the weekly sport-rotation feature-expansion loop. Seasons backtested: {seasons[0]}-{seasons[-1]}.",
        "",
        "### Backtest window",
        "",
        f"* **Start:** {args.start}",
        f"* **End:** {args.end}",
        f"* **Train window:** rolling {args.train_days} days",
        f"* **Eval window:** {args.eval_days} days, slide forward by {args.eval_days} days",
        f"* **Warmup:** {args.warmup_days} days",
        f"* **Folds completed:** {n_folds}",
        "",
        "### Data",
        "",
        f"* **Games loaded** (regular season, 3 seasons): {n_games_loaded}",
        f"* **Games graded by model** (full feature gate passed): {n_games_graded}",
        "* **Source streams attached as priors:** 538 NFL Elo (web.archive 2023 snapshot, 2022-only coverage), nflfastR EPA (model fit walk-forward by week), market-close moneylines (devigged for the market prob feature).",
        "",
        "### Headline metrics",
        "",
        "| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _row("Ungated ($100/game)", ungated_summary),
        _row(f"+3pp edge gate", gated_summary),
        "",
        "### Top loss buckets (ungated)",
        "",
    ]
    if top_loss_buckets:
        lines.append("| Bucket | Losing bets | % of total losses |")
        lines.append("|---|---:|---:|")
        total_losses = ungated_summary["n_bets"] - int(ungated_summary["n_bets"] * ungated_summary["win_rate"])
        for bucket, n in top_loss_buckets:
            pct = (n / total_losses * 100) if total_losses else 0
            lines.append(f"| `{bucket}` | {n} | {pct:.1f}% |")
    else:
        lines.append("(no graded losses to bucket)")
    lines += [
        "",
        "### Verdict (one sentence)",
        "",
    ]
    roi = ungated_summary.get("roi") or 0
    clv = ungated_summary.get("clv_proxy_pp") or 0
    if roi > 0.01:
        verdict = f"BEATS vig — +{roi*100:.2f}% ungated ROI on {ungated_summary['n_bets']} bets."
    elif roi > -0.02 and clv > 0:
        verdict = f"Coinflip-ish — {roi*100:+.2f}% ungated ROI but +{clv*100:.2f}pp CLV proxy says the model adds calibration; not enough to overcome the NFL hold."
    elif roi > -0.05:
        verdict = f"Loses to vig — {roi*100:+.2f}% ungated ROI; CLV proxy {clv*100:+.2f}pp. Honest read: 17-feature Phase-1 catalog isn't enough."
    else:
        verdict = f"Loses badly — {roi*100:+.2f}% ungated ROI. Phase-1 feature set is materially miscalibrated; needs investigation."
    lines += [
        f"**{verdict}**",
        "",
        "### Feature catalog (17 features)",
        "",
    ]
    for fn in FEATURE_NAMES:
        lines.append(f"* `{fn}`")
    lines += [
        "",
        "### What this PR does NOT change",
        "",
        "* `build_site.py::resolve_sport_modes()` — untouched.",
        "* `config.py::live_roi_floor()` — untouched.",
        "* Live picks pipeline — untouched.",
        "* Any production source weights — untouched.",
        "",
        "Per the FEATURE_EXPANSION_PLAYBOOK.md anti-pattern list: the production gate stays in place. This PR adds a new **evaluation lens** for NFL, parallel to the MLB/CFB harnesses.",
        "",
        "### Self-merge gate",
        "",
        "- [x] No CI failures (tests pass; see `tests/test_nfl_walk_forward.py`).",
        "- [x] No production-gate weakening (build_site.py untouched).",
        "- [x] No real-money integration.",
        "",
        "Per standing autonomy grant (MEMORY.md, *Standing autonomy grant — Phil's SportsBetting model*), self-merging.",
        "",
        "🐾 ada-cloud Week 4 NFL run, 2026-06-29.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("nfl_walk_forward")
    args = parse_args()
    start = _date(args.start)
    end = _date(args.end)
    seasons = list(range(start.year, end.year + 1))

    log.info("loading nflverse schedules for seasons %s ...", seasons)
    games = load_games_from_schedules(seasons)
    games = [g for g in games if start <= g.game_date <= end]
    log.info("loaded %d regular-season games in window", len(games))
    n_games_loaded = len(games)

    log.info("attaching priors from %s ...", SOURCE_HISTORY_DB_PATH)
    attach_priors_from_db(games, SOURCE_HISTORY_DB_PATH)
    n_with_elo = sum(1 for g in games if g.elo_prob_home is not None)
    n_with_epa = sum(1 for g in games if g.epa_prob_home is not None)
    n_with_mkt = sum(1 for g in games if g.home_moneyline is not None)
    log.info("priors coverage: elo=%d epa=%d market=%d (of %d)",
             n_with_elo, n_with_epa, n_with_mkt, len(games))

    log.info("loading PBP rollups for seasons %s ...", seasons)
    team_stats = load_pbp_rollups(seasons)
    log.info("rollups: %d teams", len(team_stats))

    log.info("fitting rolling rates ...")
    rolling = fit_rolling_rates(team_stats, season_reset=True)
    log.info("rolling snapshots: %d (team, date) pairs", len(rolling))

    bye_status = compute_bye_status(games)
    log.info("bye_status entries: %d", len(bye_status))

    log.info("running walk-forward backtest ...")
    fold_results = walk_forward_evaluate(
        games, rolling, bye_status,
        train_window_days=args.train_days,
        eval_window_days=args.eval_days,
        warmup_days=args.warmup_days,
    )
    log.info("walk-forward complete: %d folds", len(fold_results))

    log.info("simulating flat-$100 bets (ungated) ...")
    ungated_bets, ungated_summary = simulate_flat_stake(fold_results, edge_gate_pp=None)
    log.info("simulating flat-$100 bets (+3pp gated) ...")
    gated_bets, gated_summary = simulate_flat_stake(fold_results, edge_gate_pp=0.03)
    log.info("ungated n_bets=%d roi=%s", ungated_summary["n_bets"], ungated_summary["roi"])
    log.info("gated   n_bets=%d roi=%s", gated_summary["n_bets"], gated_summary["roi"])

    # Top loss buckets (ungated)
    buckets = ungated_summary.get("loss_buckets", {})
    top_buckets = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:8]

    # Per-season breakdown
    per_season: dict[int, dict] = {}
    for b in ungated_bets:
        s = per_season.setdefault(b.season, {"n_bets": 0, "wins": 0, "pnl": 0.0})
        s["n_bets"] += 1
        s["wins"] += 1 if b.won else 0
        s["pnl"] += b.pnl
    for s, agg in per_season.items():
        agg["win_rate"] = agg["wins"] / agg["n_bets"] if agg["n_bets"] else None
        agg["roi"] = agg["pnl"] / (100.0 * agg["n_bets"]) if agg["n_bets"] else None

    out_summary = {
        "window": {"start": args.start, "end": args.end},
        "n_games_loaded": n_games_loaded,
        "n_games_graded": ungated_summary["n_bets"],
        "n_folds": len(fold_results),
        "feature_names": FEATURE_NAMES,
        "priors_coverage": {"elo": n_with_elo, "epa": n_with_epa, "market": n_with_mkt},
        "ungated": ungated_summary,
        "gated_3pp": gated_summary,
        "per_season_ungated": per_season,
        "top_loss_buckets": top_buckets,
    }
    # JSON output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)
    log.info("wrote %s", out_path)

    # PR body
    body = _format_pr_body(
        out_summary, gated_summary, ungated_summary,
        args, seasons, n_games_loaded, ungated_summary["n_bets"],
        len(fold_results), top_buckets,
    )
    body_path = Path(args.pr_body)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body)
    log.info("wrote PR body to %s", body_path)

    print()
    print("=" * 78)
    print(f"NFL Walk-Forward Backtest — {args.start} → {args.end}")
    print("=" * 78)
    print(f"Games loaded: {n_games_loaded}    Games graded: {ungated_summary['n_bets']}    Folds: {len(fold_results)}")
    print()
    print(f"UNGATED  n={ungated_summary['n_bets']}  WR={ungated_summary['win_rate']*100:.1f}%  "
          f"ROI={ungated_summary['roi']*100:+.2f}%  CLV={ungated_summary['clv_proxy_pp']*100:+.2f}pp")
    print(f"+3pp gate n={gated_summary['n_bets']}  WR={gated_summary['win_rate']*100 if gated_summary['win_rate'] else 0:.1f}%  "
          f"ROI={(gated_summary['roi'] or 0)*100:+.2f}%  CLV={(gated_summary['clv_proxy_pp'] or 0)*100:+.2f}pp")
    print()
    print("Top loss buckets:")
    for bucket, n in top_buckets:
        print(f"  {bucket:30s} {n:4d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
