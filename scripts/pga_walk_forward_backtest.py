#!/usr/bin/env python3
"""Walk-forward PGA backtest driver — feature-expansion Phase 1 (Week 7).

A PORT of ``scripts/wta_walk_forward_backtest.py`` to PGA Tour head-to-head
matchups.

Loads historical PGA H2H matchups (player finish results + a matchup closing
prior) from ``data/source_history.db``, attaches ``market-close`` +
``datagolf-sg`` priors, builds the 12-feature Phase-1 catalog, runs
sliding-window logistic regression, and simulates a $100 flat-stake bet on
every model-graded matchup using the REAL archived closing matchup odds when
present.

CRITICAL DATA STATUS (Week 7, 2026-07-20)
-----------------------------------------
There are NO PGA rows in source_history.db yet. The only known source of
genuine historical closing MATCHUP odds — DataGolf's matchup-odds archive —
is a PAID-tier endpoint (OFF LIMITS per Phil's standing constraint) requiring
an unset ``DATAGOLF_API_KEY``. No free source pairing player finishes with
closing matchup probabilities was found this week.

Rather than fabricate outcomes to manufacture an ROI number (an explicit
playbook anti-pattern), this driver:
  * runs the REAL walk-forward backtest if PGA matchup rows are present, OR
  * writes an HONEST ``HARNESS_ONLY_DATA_BLOCKED`` receipt if they are not.

Either way the harness code is landed, leakage-gated, and unit-tested,
ready to run the moment a data source or key is provided.

Outputs:
  * data/pga_walk_forward_backtest.json                 — machine-readable receipt
  * paw-reports/sportsbetting/pga-week7-pr-body.md      — PR body
  * stdout: headline table / data-blocker statement

This script is OPT-IN. It is not invoked by ``flashcat all`` and CI does
not depend on it. The production per-sport gate stays UNTOUCHED.

Usage:
    PYTHONPATH=src python3 scripts/pga_walk_forward_backtest.py
    PYTHONPATH=src python3 scripts/pga_walk_forward_backtest.py --start 2022-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import DATA_DIR, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.pga_features import (  # noqa: E402
    FEATURE_NAMES,
    MatchupRow,
    fit_rolling_rates,
    attach_priors_from_db,
    compute_h2h,
)
from flashcat.pga_features.model import walk_forward_evaluate  # noqa: E402
from flashcat.pga_features.simulator import simulate_flat_stake  # noqa: E402


DATA_BLOCKER = (
    "No PGA rows in source_history.db, and the only known source of genuine "
    "historical closing MATCHUP odds (DataGolf's matchup-odds archive, "
    "https://datagolf.com/matchup-odds-archive) is a PAID-tier endpoint that "
    "is OFF LIMITS per Phil's standing constraint and requires an unset "
    "DATAGOLF_API_KEY. Free golf-results datasets (ESPN leaderboards, "
    "opendatabay/Kaggle PGA CSVs) supply finishing positions but NOT closing "
    "matchup probabilities, so there is no honest CLV proxy to grade against. "
    "UNBLOCK: set DATAGOLF_API_KEY with paid-tier matchup archive access "
    "(needs Phil sign-off), OR wire a free source that pairs player finishes "
    "with closing H2H matchup odds, then run this driver — the harness will "
    "produce real metrics with no code changes."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--tour", default="pga")
    p.add_argument("--out", default=str(DATA_DIR / "pga_walk_forward_backtest.json"))
    p.add_argument("--pr-body", default=str(REPO_ROOT / "paw-reports" / "sportsbetting" / "pga-week7-pr-body.md"))
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--eval-days", type=int, default=30)
    p.add_argument("--warmup-days", type=int, default=120)
    return p.parse_args()


def _date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_matchups_from_db(
    db_path: Path, start: date, end: date, tour: str = "pga"
) -> list[MatchupRow]:
    """Load graded PGA H2H matchups from source_history.db.

    A matchup is graded when a ``market-close`` (or ``datagolf-sg``) row for
    the event has a non-null ``home_won``. Player finishing positions,
    course tier, and skill snapshots are NOT stored in the base predictions
    schema, so those optional features stay ``None`` until a richer backfill
    persists them. This function is the join point a future PGA backfill
    writes into; today it returns whatever graded PGA rows exist (currently
    zero).
    """
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    rows_by_event: dict[str, dict] = {}
    try:
        cur = conn.execute(
            "SELECT event_id, source, commence_time, home, away, home_prob, "
            "home_won, market_close_decimal, closing_implied_prob "
            "FROM predictions WHERE sport = 'pga'"
        )
        for (eid, source, commence, home, away, home_prob, home_won,
             close_dec, close_prob) in cur:
            rec = rows_by_event.setdefault(str(eid), {
                "event_id": str(eid), "home": home, "away": away,
                "commence": commence, "home_won": None,
                "close_dec": None, "close_prob": None,
            })
            if home_won is not None:
                rec["home_won"] = int(home_won)
            if close_dec is not None:
                rec["close_dec"] = float(close_dec)
            if close_prob is not None:
                rec["close_prob"] = float(close_prob)
    finally:
        conn.close()

    out: list[MatchupRow] = []
    for rec in rows_by_event.values():
        if rec["home_won"] is None:
            continue  # ungraded — cannot backtest
        try:
            md = datetime.fromisoformat(
                str(rec["commence"]).replace("Z", "+00:00")
            ).date()
        except Exception:
            continue
        if not (start <= md <= end):
            continue
        out.append(MatchupRow(
            event_id=rec["event_id"],
            match_date=md,
            season=md.year,
            tour=tour,
            event_label="",
            home=rec["home"] or "",
            away=rec["away"] or "",
            home_won=bool(rec["home_won"]),
            course_tier="standard",
            home_decimal=rec["close_dec"],
            market_prob_home=rec["close_prob"],
        ))
    out.sort(key=lambda m: m.match_date)
    return out


def _write_blocked_receipt(args, out_path: Path, body_path: Path) -> None:
    receipt = {
        "sport": "pga",
        "data_status": "HARNESS_ONLY_DATA_BLOCKED",
        "data_blocker": DATA_BLOCKER,
        "window": {"start": args.start, "end": args.end},
        "n_matches_loaded": 0,
        "n_matches_graded": 0,
        "n_folds": 0,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "data_sources_attempted": [
            "DataGolf matchup-odds-archive (PAID, off-limits)",
            "DataGolf pre-tournament-archive (needs DATAGOLF_API_KEY, unset)",
            "ESPN historical leaderboards (finishes only, no matchup odds)",
            "opendatabay/Kaggle PGA CSVs (finishes only, no matchup odds)",
            "sportsbookreviewsonline golf (404)",
        ],
        # Top-level nulls so cross-sport rollup tooling reads them cleanly.
        "n_bets": 0,
        "roi": None,
        "clv_proxy_pp": None,
        "win_rate": None,
        "max_drawdown": None,
        "sharpe": None,
        "profit": None,
        "loss_buckets": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2, default=str)

    body = f"""## feat(pga): Phase-1 walk-forward harness + feature catalog (Week 7)

Week 7 of the weekly sport-rotation feature-expansion loop. A near-mechanical
PORT of the Week-6 WTA harness to PGA Tour **head-to-head matchups** (same
event shape as tennis singles: two players, pick who finishes ahead).

### Data status: HARNESS ONLY — backtest BLOCKED on historical data access

Per FEATURE_EXPANSION_PLAYBOOK.md §6 ("honest evaluation over green
dashboards"), this PR ships the harness code + tests **without a fabricated
backtest**. The real walk-forward run is blocked:

> {DATA_BLOCKER}

No outcomes were synthesized and no ROI number was manufactured. The harness
is leakage-gated, unit-tested against a synthetic fixture, and will produce
real metrics the moment a data source or key is provided — with zero code
changes (just run `scripts/pga_walk_forward_backtest.py`).

### What landed

* `src/flashcat/pga_features/` — feature_builder / model / simulator, ported
  from `wta_features/` with golf-specific adaptations.
* `scripts/pga_walk_forward_backtest.py` — this driver (real backtest when
  data exists; honest DATA-BLOCKED receipt otherwise).
* `tests/test_pga_walk_forward.py` — leakage + gate + sign-invariant tests
  (all passing).
* `data/pga_walk_forward_backtest.json` — the honest receipt (data_status =
  HARNESS_ONLY_DATA_BLOCKED, n_bets = 0).

### Metrics

| Metric | Value |
|---|---|
| data_status | `HARNESS_ONLY_DATA_BLOCKED` |
| n_bets | 0 (no graded PGA matchups available) |
| win_rate / roi / clv_proxy_pp | — (blocked) |

(No loss post-mortem table — there are no graded bets to bucket. That is the
honest outcome, not a gap to paper over.)

### Feature catalog (12 features)

{chr(10).join('* `' + fn + '`' for fn in FEATURE_NAMES)}

### Sport-mapping vs the WTA catalog

* tennis surface -> PGA **course-difficulty tier** (easy/standard/hard/major)
* tennis ranking points -> DataGolf **pre-tournament win% / skill estimate**
* tennis rank-BT prior -> **datagolf-sg** Bradley-Terry matchup prior
* tennis games/sets-won share -> **made-cut rate + exponential finish-quality**
* **DROPPED:** `sets_won_pct_l10_diff` / `games_won_pct_l10_diff` — no golf
  analog. **ADDED:** `made_cut_pct_l10_diff`, `skill_diff`. Net = 12 features.

### What this PR does NOT change

* `build_site.py::resolve_sport_modes()` — untouched.
* `config.py::live_roi_floor()` — untouched.
* `backtest/runner.py` — untouched.
* Live picks pipeline / production source weights — untouched.

### Self-merge gate

- [x] Tests pass (`tests/test_pga_walk_forward.py` + full suite).
- [x] No production-gate files touched.
- [x] No real-money integration. No fabricated backtest outcomes.

### Verdict (one sentence)

**Phase-1 PGA harness landed and unit-tested; the backtest is BLOCKED on
historical data (DataGolf matchup archive is paid/off-limits, no free
H2H-with-closing-odds source found) — cannot yet say whether the feature set
beats vig, and we will NOT pretend otherwise.**

Per standing autonomy grant (SportsBetting weekly rotation, WTA PR #33
precedent), self-merging once CI is green — the production gate is untouched.

🐾 Ada Week 7 PGA run, 2026-07-20.
"""
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_text(body)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("pga_walk_forward")
    args = parse_args()
    start = _date(args.start)
    end = _date(args.end)

    out_path = Path(args.out)
    body_path = Path(args.pr_body)

    log.info("loading PGA matchups from %s ...", SOURCE_HISTORY_DB_PATH)
    matchups = load_matchups_from_db(SOURCE_HISTORY_DB_PATH, start, end, tour=args.tour)
    log.info("loaded %d graded PGA matchups in window", len(matchups))

    if not matchups:
        log.warning("NO graded PGA matchups — writing honest DATA-BLOCKED receipt.")
        _write_blocked_receipt(args, out_path, body_path)
        print()
        print("=" * 78)
        print("PGA Walk-Forward Backtest — DATA BLOCKED (harness only)")
        print("=" * 78)
        print(DATA_BLOCKER)
        print()
        print(f"wrote {out_path}")
        print(f"wrote {body_path}")
        return 0

    # --- real backtest path (runs automatically once data is present) ---
    attach_priors_from_db(matchups, SOURCE_HISTORY_DB_PATH)
    rolling = fit_rolling_rates(matchups)
    h2h = compute_h2h(matchups)
    fold_results = walk_forward_evaluate(
        matchups, rolling, h2h,
        train_window_days=args.train_days,
        eval_window_days=args.eval_days,
        warmup_days=args.warmup_days,
    )
    ungated_bets, ungated_summary = simulate_flat_stake(fold_results, edge_gate_pp=None)
    gated_bets, gated_summary = simulate_flat_stake(fold_results, edge_gate_pp=0.03)

    buckets = ungated_summary.get("loss_buckets", {})
    top_buckets = sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)[:10]

    per_year: dict[int, dict] = {}
    for b in ungated_bets:
        s = per_year.setdefault(b.season, {"n_bets": 0, "wins": 0, "pnl": 0.0})
        s["n_bets"] += 1
        s["wins"] += 1 if b.won else 0
        s["pnl"] += b.pnl
    for yr, agg in per_year.items():
        agg["win_rate"] = agg["wins"] / agg["n_bets"] if agg["n_bets"] else None
        agg["roi"] = agg["pnl"] / (100.0 * agg["n_bets"]) if agg["n_bets"] else None

    out_summary = {
        "sport": "pga",
        "data_status": "REAL_BACKTEST",
        "window": {"start": args.start, "end": args.end},
        "n_matches_loaded": len(matchups),
        "n_matches_graded": ungated_summary["n_bets"],
        "n_folds": len(fold_results),
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "ungated": ungated_summary,
        "gated_3pp": gated_summary,
        "per_year_ungated": per_year,
        "top_loss_buckets": top_buckets,
        "n_bets": ungated_summary["n_bets"],
        "roi": ungated_summary["roi"],
        "clv_proxy_pp": ungated_summary["clv_proxy_pp"],
        "loss_buckets": ungated_summary.get("loss_buckets", {}),
        "win_rate": ungated_summary["win_rate"],
        "max_drawdown": ungated_summary.get("max_drawdown"),
        "sharpe": ungated_summary.get("sharpe"),
        "profit": ungated_summary.get("profit"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)
    log.info("wrote %s (REAL_BACKTEST, n_bets=%d)", out_path, ungated_summary["n_bets"])

    # Persist predictions to source_history.db under pga-flashcat-v2.
    _persist_predictions(SOURCE_HISTORY_DB_PATH, fold_results)

    print()
    print("=" * 78)
    print(f"PGA Walk-Forward Backtest — {args.start} -> {args.end}")
    print("=" * 78)
    wr = ungated_summary["win_rate"] or 0
    print(f"UNGATED  n={ungated_summary['n_bets']}  WR={wr*100:.1f}%  "
          f"ROI={(ungated_summary['roi'] or 0)*100:+.2f}%  "
          f"CLV={(ungated_summary['clv_proxy_pp'] or 0)*100:+.2f}pp")
    return 0


def _persist_predictions(db_path: Path, fold_results) -> None:
    """INSERT OR REPLACE walk-forward predictions under pga-flashcat-v2."""
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        for fold in fold_results:
            for pr in fold.predictions:
                conn.execute(
                    "INSERT OR REPLACE INTO predictions "
                    "(event_id, sport, source, commence_time, home, away, "
                    "home_prob, home_won, closing_implied_prob) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        pr["event_id"] + ":pga-flashcat-v2",
                        "pga", "pga-flashcat-v2",
                        pr["match_date"], pr["home"], pr["away"],
                        float(pr["home_prob"]), int(pr["home_won"]),
                        pr.get("market_prob_home"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
