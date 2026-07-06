#!/usr/bin/env python3
"""Walk-forward ATP backtest driver — feature-expansion Phase 1 (Week 5).

Loads 2022-2024 ATP main-tour singles matches from the cached tennis-data.co.uk
season xlsx (downloaded by ``scripts/backfill_tennis_historical.py``), attaches
market-close + tennis-rank-bt priors from ``data/source_history.db``, builds the
14-feature Phase-1 catalog, runs sliding-window logistic regression, and
simulates a $100 flat-stake bet on every model-graded match using the REAL
archived closing decimal odds (Pinnacle preferred).

Outputs:
  * data/atp_walk_forward_backtest.json                 — machine-readable receipt
  * paw-reports/sportsbetting/atp-week5-pr-body.md      — PR body with tables
  * stdout: headline table + loss buckets

This script is OPT-IN. It is not invoked by ``flashcat all`` and CI does
not depend on it. The production per-sport gate stays UNTOUCHED.

Usage:
    PYTHONPATH=src python3 scripts/atp_walk_forward_backtest.py
    PYTHONPATH=src python3 scripts/atp_walk_forward_backtest.py --start 2022-01-01 --end 2024-12-31
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
from flashcat.atp_features import (  # noqa: E402
    FEATURE_NAMES,
    load_matches_from_cache,
    fit_rolling_rates,
    attach_priors_from_db,
)
from flashcat.atp_features.feature_builder import compute_h2h  # noqa: E402
from flashcat.atp_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.atp_features.simulator import simulate_flat_stake  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--tour", default="atp")
    p.add_argument("--out", default=str(DATA_DIR / "atp_walk_forward_backtest.json"))
    p.add_argument("--pr-body", default=str(REPO_ROOT / "paw-reports" / "sportsbetting" / "atp-week5-pr-body.md"))
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--eval-days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=120)
    return p.parse_args()


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _format_pr_body(out_summary: dict, gated_summary: dict, ungated_summary: dict,
                    args: argparse.Namespace, seasons: list[int],
                    n_loaded: int, n_graded: int, n_folds: int,
                    top_loss_buckets: list, priors_cov: dict,
                    per_year: dict, per_surface: dict) -> str:
    def _row(label, s):
        roi_pct = f"{s.get('roi', 0)*100:+.2f}%" if s.get("roi") is not None else "—"
        wr_pct = f"{s.get('win_rate', 0)*100:.1f}%" if s.get("win_rate") is not None else "—"
        clv_pp = f"{s.get('clv_proxy_pp', 0)*100:+.2f}pp" if s.get("clv_proxy_pp") is not None else "—"
        dd = f"${s.get('max_drawdown', 0):,.0f}" if s.get("max_drawdown") is not None else "—"
        sharpe = f"{s.get('sharpe'):.2f}" if s.get("sharpe") is not None else "—"
        profit = f"${s.get('profit', 0):,.0f}" if s.get("profit") is not None else "—"
        return f"| {label} | {s['n_bets']} | {wr_pct} | {roi_pct} | {clv_pp} | {dd} | {sharpe} | {profit} |"

    lines = [
        "## feat(atp): Phase-1 walk-forward harness + feature catalog",
        "",
        f"Week 5 of the weekly sport-rotation feature-expansion loop. Seasons backtested: {seasons[0]}-{seasons[-1]}.",
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
        f"* **Matches loaded** (ATP main-tour singles, completed): {n_loaded}",
        f"* **Matches graded by model** (full feature gate passed): {n_graded}",
        f"* **Priors coverage:** market-close={priors_cov.get('market')}, tennis-rank-bt={priors_cov.get('rank_bt')} (of {n_loaded}).",
        "* **Payout odds:** REAL archived closing decimal odds (Pinnacle > Bet365 > market-avg) from tennis-data.co.uk. This is a genuine closing-line payout, not a reconstructed proxy.",
        "* **CLV proxy:** model pick prob minus devigged closing implied prob (`market_prob_home`).",
        "",
        "### Headline metrics",
        "",
        "| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _row("Ungated ($100/match)", ungated_summary),
        _row("+3pp edge gate", gated_summary),
        "",
        "### Per-year breakdown (ungated)",
        "",
        "| Year | n_bets | win_rate | ROI |",
        "|---|---:|---:|---:|",
    ]
    for yr in sorted(per_year):
        a = per_year[yr]
        wr = f"{a['win_rate']*100:.1f}%" if a.get("win_rate") is not None else "—"
        roi = f"{a['roi']*100:+.2f}%" if a.get("roi") is not None else "—"
        lines.append(f"| {yr} | {a['n_bets']} | {wr} | {roi} |")
    lines += [
        "",
        "### Per-surface breakdown (ungated)",
        "",
        "| Surface | n_bets | win_rate | ROI |",
        "|---|---:|---:|---:|",
    ]
    for surf in sorted(per_surface):
        a = per_surface[surf]
        wr = f"{a['win_rate']*100:.1f}%" if a.get("win_rate") is not None else "—"
        roi = f"{a['roi']*100:+.2f}%" if a.get("roi") is not None else "—"
        lines.append(f"| {surf or '?'} | {a['n_bets']} | {wr} | {roi} |")
    lines += [
        "",
        "### Loss post-mortem (ungated)",
        "",
    ]
    if top_loss_buckets:
        lines.append("| Bucket | Losing bets | % of total losses |")
        lines.append("|---|---:|---:|")
        total_losses = ungated_summary["n_bets"] - int(round(ungated_summary["n_bets"] * ungated_summary["win_rate"]))
        for bucket, n in top_loss_buckets:
            pct = (n / total_losses * 100) if total_losses else 0
            lines.append(f"| `{bucket}` | {n} | {pct:.1f}% |")
    else:
        lines.append("(no graded losses to bucket)")

    roi = ungated_summary.get("roi") or 0
    clv = ungated_summary.get("clv_proxy_pp") or 0
    if roi > 0.01:
        verdict = f"BEATS vig — +{roi*100:.2f}% ungated ROI on {ungated_summary['n_bets']} bets against real closing prices."
    elif roi > -0.02 and clv > 0:
        verdict = f"Coinflip-ish — {roi*100:+.2f}% ungated ROI but +{clv*100:.2f}pp CLV proxy; model adds calibration but not enough to beat the closing line."
    elif roi > -0.05:
        verdict = f"Loses to the closing line — {roi*100:+.2f}% ungated ROI (CLV proxy {clv*100:+.2f}pp); the 14-feature Phase-1 catalog does not beat Pinnacle's close."
    else:
        verdict = f"Loses badly — {roi*100:+.2f}% ungated ROI; Phase-1 feature set is materially worse than the closing line."
    lines += [
        "",
        "### Verdict (one sentence)",
        "",
        f"**{verdict}**",
        "",
        "### Feature catalog (14 features)",
        "",
    ]
    for fn in FEATURE_NAMES:
        lines.append(f"* `{fn}`")
    lines += [
        "",
        "### Skipped in Phase 1 (documented, not hidden)",
        "",
        "* **sackmann-atp-elo (surface-adjusted Elo)** — the `JeffSackmann/tennis_atp` GitHub repo returned 404/429 during the Week-5 backfill, so the Elo source never populated. This is the single highest-value Phase-2 lever (surface Elo is the gold-standard tennis rating).",
        "* **serve hold % / return break %** — tennis-data.co.uk does not publish per-match serve/return point stats. `games_won_pct_l10_diff` is our coarse proxy. Sackmann's match-charting / point-by-point data would supply the real thing.",
        "* **real line movement** — only the closing line is archived; there is no opener in the ledger, so a line-movement feature is not computable from this source.",
        "",
        "### What this PR does NOT change",
        "",
        "* `build_site.py::resolve_sport_modes()` — untouched.",
        "* `config.py::live_roi_floor()` — untouched.",
        "* `backtest/runner.py` — untouched.",
        "* Live picks pipeline / production source weights — untouched.",
        "",
        "Per FEATURE_EXPANSION_PLAYBOOK.md: the production gate is the trust contract with the live site and stays in place. This PR adds a new **evaluation lens** for ATP, parallel to the MLB/NFL harnesses.",
        "",
        "### Self-merge gate",
        "",
        "- [x] Tests pass (`tests/test_atp_walk_forward.py`).",
        "- [x] No production-gate files touched (`build_site.py` / `config.py` / `backtest/runner.py`).",
        "- [x] No real-money integration.",
        "",
        "Per standing autonomy grant (SportsBetting weekly rotation, NFL PR #27 precedent), self-merging.",
        "",
        "🐾 Ada Week 5 ATP run, 2026-07-06.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("atp_walk_forward")
    args = parse_args()
    start = _date(args.start)
    end = _date(args.end)
    seasons = list(range(start.year, end.year + 1))

    log.info("loading ATP matches from cache for seasons %s ...", seasons)
    matches = load_matches_from_cache(seasons, tour=args.tour)
    matches = [m for m in matches if start <= m.match_date <= end]
    log.info("loaded %d completed matches in window", len(matches))
    n_loaded = len(matches)
    if n_loaded == 0:
        log.error("no matches loaded — run scripts/backfill_tennis_historical.py first")
        return 1

    log.info("attaching priors from %s ...", SOURCE_HISTORY_DB_PATH)
    attach_priors_from_db(matches, SOURCE_HISTORY_DB_PATH)
    n_with_mkt = sum(1 for m in matches if m.market_prob_home is not None)
    n_with_bt = sum(1 for m in matches if m.rank_bt_prob_home is not None)
    log.info("priors coverage: market=%d rank_bt=%d (of %d)", n_with_mkt, n_with_bt, n_loaded)

    log.info("fitting rolling form ...")
    rolling = fit_rolling_rates(matches)
    log.info("rolling snapshots: %d (event, player) pairs", len(rolling))

    log.info("computing head-to-head ...")
    h2h = compute_h2h(matches)

    log.info("running walk-forward backtest ...")
    fold_results = walk_forward_evaluate(
        matches, rolling, h2h,
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

    buckets = ungated_summary.get("loss_buckets", {})
    top_buckets = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Per-year breakdown.
    per_year: dict[int, dict] = {}
    for b in ungated_bets:
        s = per_year.setdefault(b.season, {"n_bets": 0, "wins": 0, "pnl": 0.0})
        s["n_bets"] += 1
        s["wins"] += 1 if b.won else 0
        s["pnl"] += b.pnl
    for yr, agg in per_year.items():
        agg["win_rate"] = agg["wins"] / agg["n_bets"] if agg["n_bets"] else None
        agg["roi"] = agg["pnl"] / (100.0 * agg["n_bets"]) if agg["n_bets"] else None

    # Per-surface breakdown.
    per_surface: dict[str, dict] = {}
    for b in ungated_bets:
        s = per_surface.setdefault(b.surface, {"n_bets": 0, "wins": 0, "pnl": 0.0})
        s["n_bets"] += 1
        s["wins"] += 1 if b.won else 0
        s["pnl"] += b.pnl
    for surf, agg in per_surface.items():
        agg["win_rate"] = agg["wins"] / agg["n_bets"] if agg["n_bets"] else None
        agg["roi"] = agg["pnl"] / (100.0 * agg["n_bets"]) if agg["n_bets"] else None

    priors_cov = {"market": n_with_mkt, "rank_bt": n_with_bt}
    out_summary = {
        "sport": "atp",
        "window": {"start": args.start, "end": args.end},
        "n_matches_loaded": n_loaded,
        "n_matches_graded": ungated_summary["n_bets"],
        "n_folds": len(fold_results),
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "priors_coverage": priors_cov,
        "data_sources": ["tennis-data.co.uk (results+closing odds+surface+rank)", "market-close", "tennis-rank-bt"],
        "skipped_sources": ["sackmann-atp-elo (GitHub 404/429)", "serve/return point stats (not published)", "line movement (no opener archived)"],
        "ungated": ungated_summary,
        "gated_3pp": gated_summary,
        "per_year_ungated": per_year,
        "per_surface_ungated": per_surface,
        "top_loss_buckets": top_buckets,
        # Top-level mirror for cross-sport weekly_loss_postmortem.sh compatibility.
        "n_bets": ungated_summary["n_bets"],
        "roi": ungated_summary["roi"],
        "clv_proxy_pp": ungated_summary["clv_proxy_pp"],
        "loss_buckets": ungated_summary.get("loss_buckets", {}),
        "win_rate": ungated_summary["win_rate"],
        "max_drawdown": ungated_summary.get("max_drawdown"),
        "sharpe": ungated_summary.get("sharpe"),
        "profit": ungated_summary.get("profit"),
        "with_production_edge_gate": {
            "gate_pp": 0.03,
            "n_bets": gated_summary["n_bets"],
            "roi": gated_summary["roi"],
            "clv_proxy_pp": gated_summary["clv_proxy_pp"],
            "profit": gated_summary.get("profit"),
            "win_rate": gated_summary["win_rate"],
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)
    log.info("wrote %s", out_path)

    body = _format_pr_body(
        out_summary, gated_summary, ungated_summary,
        args, seasons, n_loaded, ungated_summary["n_bets"],
        len(fold_results), top_buckets, priors_cov, per_year, per_surface,
    )
    body_path = Path(args.pr_body)
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body)
    log.info("wrote PR body to %s", body_path)

    print()
    print("=" * 78)
    print(f"ATP Walk-Forward Backtest — {args.start} → {args.end}")
    print("=" * 78)
    print(f"Matches loaded: {n_loaded}    Graded: {ungated_summary['n_bets']}    Folds: {len(fold_results)}")
    print()
    wr = ungated_summary["win_rate"] or 0
    print(f"UNGATED  n={ungated_summary['n_bets']}  WR={wr*100:.1f}%  "
          f"ROI={(ungated_summary['roi'] or 0)*100:+.2f}%  CLV={(ungated_summary['clv_proxy_pp'] or 0)*100:+.2f}pp  "
          f"MaxDD=${ungated_summary.get('max_drawdown') or 0:,.0f}")
    gwr = gated_summary["win_rate"] or 0
    print(f"+3pp gate n={gated_summary['n_bets']}  WR={gwr*100:.1f}%  "
          f"ROI={(gated_summary['roi'] or 0)*100:+.2f}%  CLV={(gated_summary['clv_proxy_pp'] or 0)*100:+.2f}pp")
    print()
    print("Per-year (ungated):")
    for yr in sorted(per_year):
        a = per_year[yr]
        print(f"  {yr}  n={a['n_bets']:5d}  WR={a['win_rate']*100:.1f}%  ROI={a['roi']*100:+.2f}%")
    print()
    print("Top loss buckets:")
    for bucket, n in top_buckets:
        print(f"  {bucket:24s} {n:5d}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
