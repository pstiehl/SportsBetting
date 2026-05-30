"""Walk-forward hold-out validation for the blender.

This module guards against the central risk in the de-dilution PR:
*overfitting the backtest* by tuning the exclusion floor and β to maximize
backtested ROI. The protocol is intentionally rigid:

1. Split ``data/source_history.db`` per sport into a TRAINING window
   (2022-01-01 → 2023-12-31) and a HELD-OUT window (2024-01-01 →
   2024-12-31). The training window may not see the hold-out outcomes.
2. Compute each source's training-window Brier and ROI from the training
   predictions only.
3. Fit per-sport weights using the standard reweighter against those
   training-window meta rows (i.e. exclusion floor + softmax with β=16).
4. Apply those frozen weights to the held-out 2024 predictions to compute
   the blended probability per event. Score the blend on the actual 2024
   outcomes.
5. Return TRAINING-window vs HELD-OUT blended ROI per sport.

If the held-out ROI is meaningfully worse than the training-window ROI
(more than 5pp degradation), that's the overfit signature and the PR
ships with a warning instead of declaring victory. The accompanying
regression test enforces this gate for any sport with ≥200 held-out bets.

Nothing in this module mutates ``data/source_weights.json`` — the live
weights are still fit on the full rolling window. The held-out evaluation
is read-only by design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from ..config import (
    SOURCE_HISTORY_DB_PATH,
    blender_min_bets_for_exclusion,
    blender_roi_floor,
    hybrid_beta,
    hybrid_lambda,
)
from .reweight import _compute_pool_weights, min_events_for, weight_mode

# Walk-forward split.
TRAIN_START = date(2022, 1, 1)
TRAIN_END = date(2023, 12, 31)
HOLDOUT_START = date(2024, 1, 1)
HOLDOUT_END = date(2024, 12, 31)


@dataclass
class SourceWindowStats:
    sport: str
    source: str
    n_events: int
    n_bets: int
    brier: float | None
    roi: float | None
    wins: int


@dataclass
class HoldoutResult:
    sport: str
    train_roi: float | None
    train_n_bets: int
    holdout_roi: float | None
    holdout_n_bets: int
    sources_in_blend: list[str] = field(default_factory=list)
    sources_excluded: list[dict] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    top_source: str | None = None
    top_source_roi: float | None = None

    @property
    def delta_pp(self) -> float | None:
        if self.train_roi is None or self.holdout_roi is None:
            return None
        return (self.holdout_roi - self.train_roi) * 100.0


def _parse_dt(s: str) -> date:
    # ``2024-09-30T20:00:00Z`` or ``2022-01-01T00:00:00+00:00``
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).date()
    except Exception:
        return date(1900, 1, 1)


def _fetch_predictions(
    db_path: Path,
    sport: str,
    start: date,
    end: date,
) -> list[dict]:
    """Read predictions rows within ``[start, end]`` for one sport."""
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT event_id, sport, source, commence_time, home_prob, home_won,
                   market_close_decimal
            FROM predictions
            WHERE sport = ?
              AND commence_time >= ?
              AND commence_time <= ?
              AND home_won IS NOT NULL
            """,
            (sport, start.isoformat(), end.isoformat() + "T23:59:59Z"),
        ).fetchall()
    return [dict(r) for r in rows]


def _fetch_meta_rows(db_path: Path) -> list[dict]:
    """Read every persisted meta row (one per (sport, source, window_end))."""
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT sport, source, window_start, window_end, n_events, n_bets, "
            "brier, roi FROM meta"
        ).fetchall()
    return [dict(r) for r in rows]


def _meta_windowed_stats(
    meta_rows: list[dict],
    *,
    sport: str,
    train_end: date,
    holdout_end: date,
) -> tuple[dict[str, SourceWindowStats], dict[str, SourceWindowStats]]:
    """Derive per-source training-window and hold-out-window stats from meta.

    Meta rows in this repo are stored as cumulative-from-window-start, so to
    isolate the 2024 hold-out we subtract the training-window aggregates
    from the full window (linear in profit & wagered — same trick the
    online ROI accumulator uses).
    """
    train_end_iso = train_end.isoformat()
    holdout_end_iso = holdout_end.isoformat()

    # Pick the latest train-window row per source (window_end <= train_end).
    train_best: dict[str, dict] = {}
    full_best: dict[str, dict] = {}
    for r in meta_rows:
        if r.get("sport") != sport:
            continue
        we = r.get("window_end") or ""
        src = r.get("source")
        if not src:
            continue
        if we <= train_end_iso:
            cur = train_best.get(src)
            if cur is None or (we > (cur.get("window_end") or "")):
                train_best[src] = r
        if we <= holdout_end_iso:
            cur = full_best.get(src)
            if cur is None or (we > (cur.get("window_end") or "")):
                full_best[src] = r

    def _to_stats(row: dict) -> SourceWindowStats:
        return SourceWindowStats(
            sport=row.get("sport") or sport,
            source=row.get("source") or "",
            n_events=int(row.get("n_events") or 0),
            n_bets=int(row.get("n_bets") or 0),
            brier=row.get("brier"),
            roi=row.get("roi"),
            wins=0,
        )

    train_stats = {src: _to_stats(r) for src, r in train_best.items()}

    # Holdout = full - train (when both exist).
    holdout_stats: dict[str, SourceWindowStats] = {}
    for src, full_row in full_best.items():
        tr = train_best.get(src)
        f_n_bets = int(full_row.get("n_bets") or 0)
        f_n_events = int(full_row.get("n_events") or 0)
        f_roi = full_row.get("roi")
        if tr is None:
            # No training-window row — entire span is hold-out.
            holdout_stats[src] = _to_stats(full_row)
            continue
        t_n_bets = int(tr.get("n_bets") or 0)
        t_n_events = int(tr.get("n_events") or 0)
        t_roi = tr.get("roi")
        h_n_bets = f_n_bets - t_n_bets
        h_n_events = f_n_events - t_n_events
        if h_n_bets <= 0 or f_roi is None or t_roi is None:
            # Can't subtract — hold-out ROI unrecoverable for this source.
            holdout_stats[src] = SourceWindowStats(
                sport=sport, source=src,
                n_events=max(0, h_n_events), n_bets=max(0, h_n_bets),
                brier=None, roi=None, wins=0,
            )
            continue
        # ROI = profit / wagered; on $100 flat stake: profit = roi * 100 * n_bets,
        # wagered = 100 * n_bets. So h_roi = (f*f_n - t*t_n) / (f_n - t_n).
        h_roi = (float(f_roi) * f_n_bets - float(t_roi) * t_n_bets) / h_n_bets
        holdout_stats[src] = SourceWindowStats(
            sport=sport, source=src,
            n_events=h_n_events, n_bets=h_n_bets,
            brier=None, roi=h_roi, wins=0,
        )
    return train_stats, holdout_stats


def _weighted_avg_roi(
    stats: dict[str, SourceWindowStats],
    weights: dict[str, float],
) -> tuple[float | None, int]:
    """Weighted-average ROI across the in-blend sources.

    This is the *approximation* the holdout uses when per-event blending
    isn't possible (predictions table lacks market_close_decimal). It's not
    a true blended ROI — sources with disagreement get averaged rather
    than the higher-prob side winning the pick — but it's a useful upper
    bound on the de-dilution benefit because removing low-ROI sources
    *lifts* the weighted average. The flat-$100 simulator (see
    ``flat_stake_backtest``) gives the per-event picture.
    """
    if not weights or not stats:
        return None, 0
    pieces: list[tuple[float, float, int]] = []  # (weight, roi, n_bets)
    for src, w in weights.items():
        s = stats.get(src)
        if s is None or s.roi is None or s.n_bets <= 0:
            continue
        pieces.append((w, float(s.roi), s.n_bets))
    if not pieces:
        return None, 0
    w_total = sum(p[0] for p in pieces)
    if w_total <= 0:
        return None, 0
    avg_roi = sum(w * roi for w, roi, _ in pieces) / w_total
    n_bets = max(n for _, _, n in pieces)
    return avg_roi, n_bets


def _list_sports(db_path: Path) -> list[str]:
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT sport FROM predictions WHERE home_won IS NOT NULL"
        ).fetchall()
    return sorted({r[0] for r in rows})


def _per_source_stats(rows: list[dict]) -> dict[str, SourceWindowStats]:
    """Compute per-source Brier/ROI/n on a flat list of prediction rows."""
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)

    out: dict[str, SourceWindowStats] = {}
    for source, rs in by_source.items():
        if not rs:
            continue
        sport = rs[0].get("sport") or ""
        # Brier
        brier_terms: list[float] = []
        for r in rs:
            p = float(r["home_prob"])
            y = 1.0 if r["home_won"] else 0.0
            brier_terms.append((p - y) ** 2)
        brier = sum(brier_terms) / len(brier_terms) if brier_terms else None
        # ROI on the higher-prob side at market_close_decimal
        wagered = 0.0
        profit = 0.0
        wins = 0
        n_bets = 0
        for r in rs:
            dec = r.get("market_close_decimal")
            if dec is None or dec <= 1.0:
                continue
            pick_home = float(r["home_prob"]) >= 0.5
            won = (pick_home and r["home_won"]) or (
                not pick_home and not r["home_won"]
            )
            n_bets += 1
            wagered += 100.0
            if won:
                profit += 100.0 * (float(dec) - 1.0)
                wins += 1
            else:
                profit -= 100.0
        roi = (profit / wagered) if wagered > 0 else None
        out[source] = SourceWindowStats(
            sport=sport,
            source=source,
            n_events=len(rs),
            n_bets=n_bets,
            brier=brier,
            roi=roi,
            wins=wins,
        )
    return out


def _fit_weights_from_stats(
    stats: dict[str, SourceWindowStats],
    *,
    sport: str,
    beta: float,
    lam: float,
    roi_floor: float,
    min_bets_for_exclusion: int,
    mode: str,
) -> tuple[dict[str, float], list[dict]]:
    """Run the standard reweighter against a per-source stats dict.

    Returns ``(weights, excluded)`` matching the schema used by
    ``data/source_weights.json``.
    """
    rows: dict[str, dict] = {}
    for source, s in stats.items():
        rows[source] = {
            "n_events": s.n_events,
            "n_bets": s.n_bets,
            "brier": s.brier,
            "roi": s.roi,
        }
    min_n = min_events_for(sport)
    weights, excluded = _compute_pool_weights(
        rows, mode=mode, beta=beta, lam=lam, min_n=min_n,
        roi_floor=roi_floor,
        min_bets_for_exclusion=min_bets_for_exclusion,
    )
    return weights, excluded


def _blended_roi(
    rows: list[dict],
    weights: dict[str, float],
) -> tuple[float | None, int, int]:
    """Score a flat list of per-(event, source) predictions against weights.

    Groups by event_id, blends ``home_prob`` using ``weights`` (renormalized
    over present sources for that event), picks the higher-prob side, settles
    at ``market_close_decimal``. Returns ``(roi, n_bets, n_wins)``.
    """
    by_event: dict[str, list[dict]] = {}
    for r in rows:
        if r["source"] not in weights:
            continue
        by_event.setdefault(r["event_id"], []).append(r)

    wagered = 0.0
    profit = 0.0
    wins = 0
    n_bets = 0
    for event_id, rs in by_event.items():
        # Pull a settlement price + outcome from the first row that has one.
        dec = None
        home_won = None
        for r in rs:
            if r.get("market_close_decimal") is not None and dec is None:
                dec = float(r["market_close_decimal"])
            if r.get("home_won") is not None and home_won is None:
                home_won = bool(r["home_won"])
        if dec is None or dec <= 1.0 or home_won is None:
            continue
        # Renormalize weights over the sources actually present for this event.
        present_w = {r["source"]: weights[r["source"]] for r in rs}
        total = sum(present_w.values())
        if total <= 0:
            continue
        norm = {k: v / total for k, v in present_w.items()}
        blended = sum(float(r["home_prob"]) * norm[r["source"]] for r in rs)
        blended = max(0.0, min(1.0, blended))
        pick_home = blended >= 0.5
        won = (pick_home and home_won) or (not pick_home and not home_won)
        n_bets += 1
        wagered += 100.0
        if won:
            profit += 100.0 * (dec - 1.0)
            wins += 1
        else:
            profit -= 100.0
    if wagered <= 0:
        return None, 0, 0
    return profit / wagered, n_bets, wins


def run_holdout_validation(
    db_path: Path | None = None,
    *,
    beta: float | None = None,
    lam: float | None = None,
    roi_floor: float | None = None,
    min_bets_for_exclusion: int | None = None,
    mode: str | None = None,
    train_start: date = TRAIN_START,
    train_end: date = TRAIN_END,
    holdout_start: date = HOLDOUT_START,
    holdout_end: date = HOLDOUT_END,
) -> dict[str, HoldoutResult]:
    """Run the walk-forward hold-out validation for every sport with data.

    Returns ``{sport: HoldoutResult}``.
    """
    db = db_path or SOURCE_HISTORY_DB_PATH
    if not Path(db).exists():
        return {}
    try:
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            conn.execute("SELECT 1 FROM predictions LIMIT 1").fetchone()
    except Exception:
        return {}

    beta = beta if beta is not None else hybrid_beta()
    lam = lam if lam is not None else hybrid_lambda()
    roi_floor = (
        roi_floor if roi_floor is not None else blender_roi_floor()
    )
    min_bets_for_exclusion = (
        min_bets_for_exclusion
        if min_bets_for_exclusion is not None
        else blender_min_bets_for_exclusion()
    )
    mode = mode or weight_mode()

    # All sports the meta table knows about (some have no predictions rows).
    meta_rows = _fetch_meta_rows(db)
    sports = sorted({r["sport"] for r in meta_rows if r.get("sport")})
    results: dict[str, HoldoutResult] = {}
    for sport in sports:
        train_rows = _fetch_predictions(db, sport, train_start, train_end)
        hold_rows = _fetch_predictions(db, sport, holdout_start, holdout_end)
        train_stats_pred: dict[str, SourceWindowStats] = {}
        # Stamp the sport (rows from sqlite don't carry it back).
        if train_rows:
            train_stats_pred = _per_source_stats(train_rows)
            for s in train_stats_pred.values():
                s.sport = sport

        # Prefer per-prediction stats; otherwise fall back to meta rows.
        train_stats_meta, holdout_stats_meta = _meta_windowed_stats(
            meta_rows, sport=sport, train_end=train_end, holdout_end=holdout_end,
        )
        train_stats = {**train_stats_meta, **train_stats_pred}
        # If prediction-derived sources have ROI=None (no market_close_decimal),
        # back-fill from meta.
        for src, s in train_stats.items():
            if s.roi is None and src in train_stats_meta and train_stats_meta[src].roi is not None:
                s.roi = train_stats_meta[src].roi
                if s.n_bets == 0:
                    s.n_bets = train_stats_meta[src].n_bets

        if not train_stats:
            continue

        weights, excluded = _fit_weights_from_stats(
            train_stats, sport=sport, beta=beta, lam=lam,
            roi_floor=roi_floor,
            min_bets_for_exclusion=min_bets_for_exclusion,
            mode=mode,
        )
        if not weights:
            results[sport] = HoldoutResult(
                sport=sport,
                train_roi=None, train_n_bets=0,
                holdout_roi=None, holdout_n_bets=0,
                sources_in_blend=[],
                sources_excluded=excluded,
                weights={},
            )
            continue

        # Score: prefer per-event re-blending if predictions carry settlement
        # prices; otherwise fall back to weighted-average per-source ROI.
        train_roi: float | None
        hold_roi: float | None
        train_n: int
        hold_n: int
        if train_rows and any(
            r.get("market_close_decimal") is not None for r in train_rows
        ):
            train_roi, train_n, _ = _blended_roi(train_rows, weights)
            hold_roi, hold_n, _ = (
                _blended_roi(hold_rows, weights) if hold_rows else (None, 0, 0)
            )
        else:
            train_roi, train_n = _weighted_avg_roi(train_stats, weights)
            hold_roi, hold_n = _weighted_avg_roi(holdout_stats_meta, weights)

        # Top source by training ROI within the blend.
        top_src = None
        top_roi = None
        for src in weights:
            s = train_stats.get(src)
            if s is None or s.roi is None:
                continue
            if top_roi is None or s.roi > top_roi:
                top_roi = s.roi
                top_src = src
        results[sport] = HoldoutResult(
            sport=sport,
            train_roi=train_roi,
            train_n_bets=train_n,
            holdout_roi=hold_roi,
            holdout_n_bets=hold_n,
            sources_in_blend=sorted(weights.keys()),
            sources_excluded=excluded,
            weights=weights,
            top_source=top_src,
            top_source_roi=top_roi,
        )
    return results


def format_holdout_table(results: dict[str, HoldoutResult]) -> str:
    """Render the training vs hold-out ROI table for the PR body / docs."""
    if not results:
        return "(no holdout data — source_history.db has no graded predictions)"
    header = (
        f"{'Sport':<6}{'Sources':>9}{'Excluded':>10}"
        f"{'Train ROI':>13}{'Train N':>10}"
        f"{'Holdout ROI':>15}{'Holdout N':>12}"
        f"{'Delta (pp)':>13}"
    )
    lines = [header, "-" * len(header)]
    for sport in sorted(results.keys()):
        r = results[sport]
        n_in = len(r.sources_in_blend)
        n_ex = sum(1 for e in r.sources_excluded if e.get("source") is not None)
        tr = f"{r.train_roi*100:+.2f}%" if r.train_roi is not None else "n/a"
        hr = f"{r.holdout_roi*100:+.2f}%" if r.holdout_roi is not None else "n/a"
        d = f"{r.delta_pp:+.2f}" if r.delta_pp is not None else "n/a"
        lines.append(
            f"{sport.upper():<6}{n_in:>9}{n_ex:>10}"
            f"{tr:>13}{r.train_n_bets:>10}"
            f"{hr:>15}{r.holdout_n_bets:>12}"
            f"{d:>13}"
        )
    return "\n".join(lines)
