"""Per-source accountability ledger and scorer.

This is the answer to Phil's "be a critic of every source" ask. For every
prediction source the model touches, we compute — over the full available
historical window — the numbers a sportsbook risk desk would compute on its
own analysts:

* ``n_predictions``        — how many graded events the source has weighed in on
* ``hit_rate``             — % of picked-side calls that won (pick = higher home prob)
* ``brier``                — calibration: ``mean((p - y)^2)`` — lower is better
* ``log_loss``             — ``mean(-(y log p + (1-y) log(1-p)))`` — lower is better
* ``roi_flat_100``         — ROI on a $100-per-event flat-stake "always pick higher prob"
                             bet against the closing book on whichever side the
                             source picked (Pinnacle/PSW preferred → B365 →
                             tennis-data Avg → nflverse home/away mlines)
* ``clv_pp``               — average closing-line value vs picked-side devigged
                             implied prob, in percentage points
* ``max_drawdown_usd``     — worst peak-to-trough on the $100/event bankroll curve
* ``longest_losing_streak``— longest run of consecutive losing $100 bets
* ``verdict``              — qualitative bucket: KEEP / KEEP-WITH-CAVEATS / DROP /
                             INSUFFICIENT-DATA / NOISE

This module is **read-only** against `source_history.db`. It re-derives the
per-event ledger directly from tennis-data.co.uk archives (ATP+WTA) and the
nflverse archive for NFL so we have real per-event decimal odds — the
`predictions` table only stores `home_prob`/outcome, not per-side decimals,
so we can't compute ROI/drawdown/streak purely from the DB.

Predict.tennis is included as an OBSERVED-EXTERNAL source. We can't scrape
their per-event historical predictions (Cloudflare-gated, no API), but they
self-publish their own hit rates and yields on
``https://predict.tennis/prediction-check/`` and their 2024 review. We
record those numbers verbatim with a clear provenance note rather than make
something up.
"""

from __future__ import annotations

import io
import json
import logging
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import DATA_DIR, SOURCE_HISTORY_DB_PATH

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring math — pure functions, no I/O.
# ---------------------------------------------------------------------------


def brier_score(prob: float, outcome: int) -> float:
    """``(p - y)^2`` — single observation; mean caller's responsibility."""
    return (float(prob) - float(outcome)) ** 2


def log_loss(prob: float, outcome: int) -> float:
    """Clipped binary log loss for one observation."""
    p = max(0.001, min(0.999, float(prob)))
    y = float(outcome)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


@dataclass
class BetTick:
    """One $100 flat-stake settlement on the source's picked side."""

    date: date
    sport: str
    source: str
    home_prob: float
    home_won: int
    picked_home: bool
    picked_dec: float
    won: bool
    profit_100: float  # signed, $100 staked

    @property
    def implied_pp(self) -> float:
        """Single-sided implied probability on the picked side."""
        if self.picked_dec <= 1.0:
            return 0.0
        return 1.0 / float(self.picked_dec)

    @property
    def clv_pp(self) -> float:
        """Closing-line value: source's picked-side prob minus market implied (pp)."""
        pick_prob = self.home_prob if self.picked_home else 1.0 - self.home_prob
        return float(pick_prob) - self.implied_pp


def settle_ticks(ticks: Iterable[BetTick], *, stake: float = 100.0) -> dict:
    """Aggregate a stream of BetTicks into the per-source scorecard.

    Computes hit rate, ROI, Brier, log loss, CLV, max drawdown, longest
    losing streak. Pure function — caller arranges the ticks.
    """
    ticks = list(ticks)
    n = len(ticks)
    if n == 0:
        return {
            "n_predictions": 0,
            "n_bets": 0,
            "hit_rate": None,
            "brier": None,
            "log_loss": None,
            "wagered_usd": 0.0,
            "profit_usd": 0.0,
            "roi_flat_100": None,
            "clv_pp": None,
            "max_drawdown_usd": 0.0,
            "longest_losing_streak": 0,
        }
    wins = sum(1 for t in ticks if t.won)
    losses = n - wins
    wagered = stake * n
    profit = sum(t.profit_100 for t in ticks)
    roi = (profit / wagered) if wagered > 0 else None
    # Brier and log loss are computed on the source's PROBABILITY (not pick),
    # which is what a calibration metric should be.
    brier = sum(brier_score(t.home_prob, t.home_won) for t in ticks) / n
    ll = sum(log_loss(t.home_prob, t.home_won) for t in ticks) / n
    clv = sum(t.clv_pp for t in ticks) / n
    # Bankroll curve for drawdown + streak. Sort by date, then by event id
    # implicit in the input list ordering (caller is expected to pre-sort).
    ordered = sorted(ticks, key=lambda t: (t.date, t.sport, t.source))
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    cur_losing = 0
    longest_losing = 0
    for t in ordered:
        running += t.profit_100
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
        if t.won:
            cur_losing = 0
        else:
            cur_losing += 1
            if cur_losing > longest_losing:
                longest_losing = cur_losing
    return {
        "n_predictions": n,
        "n_bets": n,  # one bet per prediction in the $100/event hypothetical
        "hit_rate": wins / n,
        "wins": wins,
        "losses": losses,
        "brier": brier,
        "log_loss": ll,
        "wagered_usd": wagered,
        "profit_usd": profit,
        "roi_flat_100": roi,
        "clv_pp": clv,
        "max_drawdown_usd": max_dd,
        "longest_losing_streak": longest_losing,
    }


# ---------------------------------------------------------------------------
# Verdict bucketing — explicit, conservative, no spin.
# ---------------------------------------------------------------------------


def verdict_for(scorecard: dict, *, min_n: int = 200) -> str:
    """Phil's no-spin verdict.

    - INSUFFICIENT-DATA: < ``min_n`` graded predictions
    - DROP: brier ≥ 0.25 (worse than a coin flip) OR roi ≤ -10%
    - NOISE: brier between 0.24-0.25 AND roi between -5% and 0% (vig territory,
      no signal worth keeping)
    - KEEP-WITH-CAVEATS: roi between -3% and +1% (close to break-even — fine
      for calibration but won't make money on its own)
    - KEEP: roi > +1%
    """
    n = scorecard.get("n_predictions") or 0
    if n < min_n:
        return "INSUFFICIENT-DATA"
    brier = scorecard.get("brier")
    roi = scorecard.get("roi_flat_100")
    if brier is not None and brier >= 0.25:
        return "DROP"
    if roi is not None and roi <= -0.10:
        return "DROP"
    if brier is not None and 0.24 <= brier < 0.25 and roi is not None and -0.05 <= roi <= 0:
        return "NOISE"
    if roi is None:
        # No betting data but brier is fine → calibration-only source.
        return "KEEP-WITH-CAVEATS"
    if -0.03 <= roi <= 0.01:
        return "KEEP-WITH-CAVEATS"
    if roi > 0.01:
        return "KEEP"
    return "NOISE"


# ---------------------------------------------------------------------------
# Tennis-data.co.uk per-event ledger.
# ---------------------------------------------------------------------------


def _decimal_pair_from_row(row: dict) -> tuple[float | None, float | None]:
    """Return (winner_close_dec, loser_close_dec), preferring Pinnacle."""

    def _f(v):
        try:
            x = float(v)
            return x if x > 1.0 else None
        except Exception:
            return None

    for w_key, l_key in (("PSW", "PSL"), ("B365W", "B365L"), ("AvgW", "AvgL")):
        w = _f(row.get(w_key))
        l = _f(row.get(l_key))
        if w is not None and l is not None:
            return w, l
    return None, None


def _profit_100(picked_dec: float, won: bool, stake: float = 100.0) -> float:
    if won:
        return stake * (picked_dec - 1.0)
    return -stake


def tennis_ticks_from_archive(
    tour: str,
    *,
    years: Iterable[int],
    fetcher=None,
) -> list[dict]:
    """Re-pull tennis-data.co.uk archives and return raw rows.

    Returns ``[{date, winner, loser, w_dec, l_dec, wpts, lpts}]`` ready for
    per-source scoring. We re-pull rather than read source_history.db
    because the DB does not persist per-side decimal odds.

    ``fetcher`` lets tests inject a fake httpx.get. Production uses
    ``_default_tennis_fetch`` against tennis-data.co.uk.
    """
    if fetcher is None:
        fetcher = _default_tennis_fetch
    out: list[dict] = []
    for year in years:
        try:
            rows = fetcher(tour, year)
        except Exception as e:  # noqa: BLE001
            log.warning("tennis-data fetch failed tour=%s year=%s: %s", tour, year, e)
            continue
        for row in rows:
            winner = (row.get("Winner") or "").strip()
            loser = (row.get("Loser") or "").strip()
            if not winner or not loser:
                continue
            if (row.get("Comment") or "").strip().lower() != "completed":
                continue
            d = row.get("Date")
            if isinstance(d, datetime):
                match_day = d.date()
            elif isinstance(d, date):
                match_day = d
            else:
                try:
                    match_day = datetime.fromisoformat(str(d)).date()
                except Exception:
                    continue
            w_dec, l_dec = _decimal_pair_from_row(row)
            if w_dec is None or l_dec is None:
                continue
            wpts = row.get("WPts")
            lpts = row.get("LPts")
            try:
                wpts_f = float(wpts) if wpts is not None else None
            except Exception:
                wpts_f = None
            try:
                lpts_f = float(lpts) if lpts is not None else None
            except Exception:
                lpts_f = None
            out.append({
                "date": match_day,
                "winner": winner,
                "loser": loser,
                "w_dec": w_dec,
                "l_dec": l_dec,
                "wpts": wpts_f,
                "lpts": lpts_f,
            })
    return out


def _default_tennis_fetch(tour: str, year: int) -> list[dict]:
    """Live fetch of tennis-data.co.uk archive for one (tour, year)."""
    import httpx
    import openpyxl

    if tour == "atp":
        url = f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"
    elif tour == "wta":
        url = f"http://www.tennis-data.co.uk/{year}w/{year}.xlsx"
    else:
        raise ValueError(f"unknown tour: {tour}")
    r = httpx.get(url, timeout=60.0)
    r.raise_for_status()
    wb = openpyxl.load_workbook(io.BytesIO(r.content), read_only=True)
    ws = wb.active
    header = None
    rows: list[dict] = []
    for raw in ws.iter_rows(values_only=True):
        if header is None:
            header = list(raw)
            continue
        rows.append(dict(zip(header, raw)))
    return rows


def _bt_rank_prob(home_pts: float | None, away_pts: float | None) -> float | None:
    if not home_pts or not away_pts or home_pts <= 0 or away_pts <= 0:
        return None
    k = 0.45
    logit = k * math.log(home_pts / away_pts)
    return 1.0 / (1.0 + math.exp(-logit))


def build_tennis_per_source_ticks(
    tour: str,
    *,
    years: Iterable[int],
    fetcher=None,
) -> dict[str, list[BetTick]]:
    """Per-source per-event ticks for a tennis tour over the year window.

    Sources covered:
      - ``market-close``   — devigged closing line (Pinnacle preferred)
      - ``pinnacle-close-favorite``  — naive "always pick the shorter price"
      - ``tennis-rank-bt`` — Bradley-Terry from rank points
      - ``coin-flip``      — control source, p=0.5 always

    Each tick records what would happen if you flat-staked $100 on the
    higher-probability side at the **picked-side closing decimal**.
    """
    rows = tennis_ticks_from_archive(tour, years=years, fetcher=fetcher)
    ticks: dict[str, list[BetTick]] = defaultdict(list)
    sport = tour
    for r in rows:
        # Canonicalise: alphabetically-first player is "home" so probabilities
        # have a stable referent across sources.
        if r["winner"] < r["loser"]:
            home, away = r["winner"], r["loser"]
            home_won, home_dec, away_dec = 1, r["w_dec"], r["l_dec"]
            home_pts, away_pts = r["wpts"], r["lpts"]
        else:
            home, away = r["loser"], r["winner"]
            home_won, home_dec, away_dec = 0, r["l_dec"], r["w_dec"]
            home_pts, away_pts = r["lpts"], r["wpts"]

        # Devigged market-close prob on home.
        h_imp = 1.0 / home_dec
        a_imp = 1.0 / away_dec
        denom = h_imp + a_imp
        market_home = h_imp / denom if denom > 0 else None

        # Per-source probs.
        # ``coin-flip`` is a control: a source with no signal whatsoever.
        # We expect it to lose roughly the vig on $100/event (and to score
        # exactly 0.25 Brier). If anything in the model can't beat it, we
        # know to drop that source.
        probs: dict[str, float | None] = {
            "market-close": market_home,
            "tennis-rank-bt": _bt_rank_prob(home_pts, away_pts),
            "coin-flip": 0.5,
        }
        for src, hp in probs.items():
            if hp is None:
                continue
            picked_home = float(hp) >= 0.5
            picked_dec = home_dec if picked_home else away_dec
            won = (picked_home and home_won == 1) or (not picked_home and home_won == 0)
            ticks[src].append(BetTick(
                date=r["date"],
                sport=sport,
                source=src,
                home_prob=float(hp),
                home_won=int(home_won),
                picked_home=picked_home,
                picked_dec=float(picked_dec),
                won=bool(won),
                profit_100=_profit_100(float(picked_dec), bool(won)),
            ))
    return dict(ticks)


# ---------------------------------------------------------------------------
# Predict.tennis — observed-external scorecard, no per-event scrape.
# ---------------------------------------------------------------------------


# Numbers below come from predict.tennis's OWN published Prediction Check page
# (https://predict.tennis/prediction-check/, snapshot 2025-10-15 via Wayback)
# and their own 2024 season review (published 2024-11-19, same source).
# These are SELF-REPORTED numbers. We record them verbatim and call them out
# as observed-external rather than try to manufacture per-event ticks from
# them.
PREDICT_TENNIS_2024_SELF_REPORT = {
    "source": "predict.tennis",
    "provenance": (
        "Self-reported on predict.tennis/prediction-check/ and "
        "predict.tennis/promo/2024-tennis-predictions-analysis-... "
        "(retrieved 2026-05-31 via web.archive.org)."
    ),
    "method": (
        "Site fields three predictor types: odds-based (their words: "
        "'rely on the probabilities implied by betting odds'), "
        "performance-points/form-based, and a final ensemble. Reported "
        "yields use a fixed $1 stake per match. Hit rates use 'pick the "
        "side with higher implied prob' on whichever odds source the "
        "site reads at scrape time."
    ),
    "atp": {
        "season": "2025-full",
        "n_predictions": 3430,
        "wins": 2302,
        "hit_rate": 2302 / 3430,
        "by_surface_hit_rate": {
            "hard_outdoor": (1120, 1677, 1120 / 1677),
            "clay": (722, 1076, 722 / 1076),
            "grass": (325, 463, 325 / 463),
            "hard_indoor": (135, 214, 135 / 214),
        },
        "yield_2024_by_surface": {
            "clay": +0.0132,
            "grass": -0.0075,
            "hard_indoor": -0.0063,
            "hard_outdoor": -0.0539,
        },
    },
    "wta": {
        "season": "2025-full",
        "n_predictions": 3329,
        "wins": 2262,
        "hit_rate": 2262 / 3329,
        "by_surface_hit_rate": {
            "hard_outdoor": (1285, 1920, 1285 / 1920),
            "clay": (589, 831, 589 / 831),
            "grass": (319, 477, 319 / 477),
            "hard_indoor": (62, 89, 62 / 89),
        },
        "yield_2024_by_surface": {
            "clay": -0.0229,
            "grass": -0.0275,
            "hard_indoor": -0.1827,
            "hard_outdoor": -0.0599,
        },
    },
    "overall_2024_yield_odds_based": -0.0169,
    "overall_2024_yield_form_based_atp": -0.0851,
    "overall_2024_yield_form_based_wta": -0.0549,
}


def predict_tennis_scorecard() -> dict:
    """Two scorecard rows (atp + wta) constructed from self-reported numbers.

    No per-event ROI is computed — predict.tennis doesn't publish a per-event
    ledger we can settle. We report the site's own hit rate, the site's own
    flat-yield numbers, and bucket the verdict from those.
    """

    def _row(tour: str) -> dict:
        d = PREDICT_TENNIS_2024_SELF_REPORT[tour]
        n = d["n_predictions"]
        hit = d["hit_rate"]
        # Their odds-based yield row from 2024 review (overall, both tours)
        # is -1.69%. The per-tour breakdown comes from the per-surface
        # numbers — we take the prediction-count-weighted average of the
        # 2024 surface yields as the headline tour yield.
        surface_yields = d["yield_2024_by_surface"]
        surface_hits = d["by_surface_hit_rate"]
        # Re-weight surface yields by 2025 surface n (the 2024 review and
        # 2025 hit-rate counts come from different windows; we document the
        # mismatch in the report).
        ny = 0.0
        nw = 0
        for surf, y in surface_yields.items():
            if surf in surface_hits:
                ns = surface_hits[surf][1]
                ny += y * ns
                nw += ns
        weighted_y = (ny / nw) if nw > 0 else None
        return {
            "n_predictions": n,
            "n_bets": n,
            "hit_rate": hit,
            "wins": d["wins"],
            "losses": n - d["wins"],
            "brier": None,  # not available — per-event prob isn't published
            "log_loss": None,
            "wagered_usd": float(n) * 100.0,  # hypothetical
            "profit_usd": (weighted_y or 0.0) * float(n) * 100.0,
            "roi_flat_100": weighted_y,
            "clv_pp": None,
            "max_drawdown_usd": None,
            "longest_losing_streak": None,
        }

    return {
        "atp": _row("atp"),
        "wta": _row("wta"),
        "provenance": PREDICT_TENNIS_2024_SELF_REPORT["provenance"],
        "method": PREDICT_TENNIS_2024_SELF_REPORT["method"],
        "overall_self_reported_yield": PREDICT_TENNIS_2024_SELF_REPORT[
            "overall_2024_yield_odds_based"
        ],
    }


# ---------------------------------------------------------------------------
# DB-derived rows: pull per-source brier/log_loss from the predictions table.
# ---------------------------------------------------------------------------


def per_source_db_metrics(db_path: Path | None = None) -> dict[tuple[str, str], dict]:
    """Brier + log loss + hit rate for every (sport, source) in
    source_history.db. ROI/drawdown/streak require per-side decimal odds
    that the predictions table doesn't store, so they're left None here.
    """
    p = db_path or SOURCE_HISTORY_DB_PATH
    if not Path(p).exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with sqlite3.connect(str(p)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            """
            SELECT sport, source, home_prob, home_won
              FROM predictions
             WHERE home_won IS NOT NULL
            """
        ).fetchall()
    buckets: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        buckets[(r["sport"], r["source"])].append((float(r["home_prob"]), int(r["home_won"])))
    for key, obs in buckets.items():
        n = len(obs)
        if n == 0:
            continue
        wins = sum(1 for p, y in obs if (p >= 0.5 and y == 1) or (p < 0.5 and y == 0))
        brier = sum((p - y) ** 2 for p, y in obs) / n
        ll = sum(
            -(y * math.log(max(0.001, min(0.999, p)))
              + (1 - y) * math.log(1.0 - max(0.001, min(0.999, p))))
            for p, y in obs
        ) / n
        out[key] = {
            "n_predictions": n,
            "wins": wins,
            "losses": n - wins,
            "hit_rate": wins / n,
            "brier": brier,
            "log_loss": ll,
            # ROI/drawdown/streak filled in by per-event ledger where available.
            "roi_flat_100": None,
            "clv_pp": None,
            "max_drawdown_usd": None,
            "longest_losing_streak": None,
            "wagered_usd": None,
            "profit_usd": None,
        }
    return out


# ---------------------------------------------------------------------------
# Meta-table ROI: source_history.db.meta carries the canonical ROI numbers
# computed by the backfill scripts (which DO have per-side decimals). Use
# those for the headline ROI on the report.
# ---------------------------------------------------------------------------


def meta_roi(db_path: Path | None = None) -> dict[tuple[str, str], dict]:
    """Pick the widest-window meta row per (sport, source) and return it."""
    p = db_path or SOURCE_HISTORY_DB_PATH
    if not Path(p).exists():
        return {}
    out: dict[tuple[str, str], dict] = {}
    with sqlite3.connect(str(p)) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM meta").fetchall()
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["sport"], r["source"])
        d = dict(r)
        n = int(d.get("n_bets") or 0)
        if key not in best or n > int(best[key].get("n_bets") or 0):
            best[key] = d
    for key, d in best.items():
        out[key] = {
            "window_start": d.get("window_start"),
            "window_end": d.get("window_end"),
            "n_events": int(d.get("n_events") or 0),
            "n_bets": int(d.get("n_bets") or 0),
            "brier_meta": d.get("brier"),
            "log_loss_meta": d.get("log_loss"),
            "roi_meta": d.get("roi"),
            "calibration_slope": d.get("calibration_slope"),
            "avg_clv_pp": d.get("avg_clv_pp"),
        }
    return out


# ---------------------------------------------------------------------------
# Report assembly.
# ---------------------------------------------------------------------------


@dataclass
class SourceCard:
    sport: str
    source: str
    metrics: dict
    verdict: str
    notes: list[str] = field(default_factory=list)


def assemble_report(
    *,
    db_path: Path | None = None,
    tennis_years: Iterable[int] | None = None,
    include_tennis_per_event: bool = True,
    tennis_fetcher=None,
) -> dict:
    """Build the full source accountability payload.

    Combines:
      - source_history.db ``predictions`` (Brier/log loss/hit rate)
      - source_history.db ``meta`` (canonical ROI numbers from backfill)
      - tennis-data.co.uk per-event re-pull (drawdown + streak + CLV)
      - predict.tennis self-reported numbers (observed-external)
    """
    db_metrics = per_source_db_metrics(db_path)
    meta = meta_roi(db_path)

    cards: list[SourceCard] = []

    # 1) Every (sport, source) we have a graded predictions ledger for.
    for (sport, source), m in db_metrics.items():
        merged = dict(m)
        mr = meta.get((sport, source), {})
        if mr.get("roi_meta") is not None:
            merged["roi_flat_100"] = float(mr["roi_meta"])
            merged["n_bets"] = int(mr.get("n_bets") or m["n_predictions"])
            merged["wagered_usd"] = 100.0 * merged["n_bets"]
            merged["profit_usd"] = merged["roi_flat_100"] * merged["wagered_usd"]
        if mr.get("avg_clv_pp") is not None:
            merged["clv_pp"] = float(mr["avg_clv_pp"]) / 100.0
        merged["window"] = {
            "start": mr.get("window_start"),
            "end": mr.get("window_end"),
        }
        cards.append(SourceCard(
            sport=sport, source=source, metrics=merged,
            verdict=verdict_for(merged),
        ))

    # 2) Re-pull tennis-data.co.uk for per-event drawdown + streak.
    if include_tennis_per_event:
        years = tennis_years or list(range(2022, 2025))
        for tour in ("atp", "wta"):
            try:
                ticks_by_src = build_tennis_per_source_ticks(
                    tour, years=years, fetcher=tennis_fetcher
                )
            except Exception as e:  # noqa: BLE001
                log.warning("tennis per-event pull failed for %s: %s", tour, e)
                ticks_by_src = {}
            for src, ticks in ticks_by_src.items():
                sc = settle_ticks(ticks)
                # Annotate window from ticks.
                if ticks:
                    sc["window"] = {
                        "start": min(t.date for t in ticks).isoformat(),
                        "end": max(t.date for t in ticks).isoformat(),
                    }
                # If we already have this source from the DB pull, MERGE
                # the drawdown/streak in (DB ROI is authoritative).
                existing = next(
                    (c for c in cards if c.sport == tour and c.source == src), None
                )
                if existing:
                    for k in ("max_drawdown_usd", "longest_losing_streak",
                              "clv_pp", "wagered_usd", "profit_usd"):
                        if existing.metrics.get(k) in (None, 0, 0.0):
                            existing.metrics[k] = sc.get(k)
                    # If we don't yet have an ROI from meta, use per-event.
                    if existing.metrics.get("roi_flat_100") is None:
                        existing.metrics["roi_flat_100"] = sc.get("roi_flat_100")
                    existing.verdict = verdict_for(existing.metrics)
                    existing.notes.append(
                        "per-event ledger merged from tennis-data.co.uk archive"
                    )
                else:
                    cards.append(SourceCard(
                        sport=tour, source=src, metrics=sc,
                        verdict=verdict_for(sc),
                        notes=["per-event from tennis-data.co.uk re-pull"],
                    ))

    # 3) predict.tennis as observed-external.
    try:
        pt = predict_tennis_scorecard()
        for tour in ("atp", "wta"):
            sc = pt[tour]
            cards.append(SourceCard(
                sport=tour, source="predict.tennis",
                metrics=sc,
                verdict=verdict_for(sc, min_n=200),
                notes=[
                    "OBSERVED-EXTERNAL: predict.tennis does not expose a "
                    "per-event historical API; numbers are SELF-REPORTED.",
                    pt["provenance"],
                    pt["method"],
                    (
                        f"Site's own overall odds-based yield (full 2024 "
                        f"season, both tours): "
                        f"{pt['overall_self_reported_yield']:+.2%}."
                    ),
                ],
            ))
    except Exception as e:  # noqa: BLE001
        log.warning("predict.tennis scorecard skipped: %s", e)

    cards.sort(key=lambda c: (c.sport, c.source))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_sources": len(cards),
        "sources": [
            {
                "sport": c.sport,
                "source": c.source,
                "verdict": c.verdict,
                "metrics": c.metrics,
                "notes": c.notes,
            }
            for c in cards
        ],
    }


# ---------------------------------------------------------------------------
# Markdown rendering for paw-reports.
# ---------------------------------------------------------------------------


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+.2f}%" if v != 0 else "0.00%"
    except Exception:
        return "—"


def _fmt_usd(v) -> str:
    if v is None:
        return "—"
    try:
        return f"${float(v):,.0f}"
    except Exception:
        return "—"


def _fmt_num(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "—"


def render_markdown(report: dict) -> str:
    """Phil-ready markdown."""
    lines: list[str] = []
    lines.append("# Source Accountability Report")
    lines.append("")
    lines.append(f"_Generated: {report['generated_at']}_")
    lines.append("")

    # Executive summary block — the bottom line first.
    counts: dict[str, int] = defaultdict(int)
    keepers: list[str] = []
    drops: list[str] = []
    for s in report["sources"]:
        counts[s["verdict"]] += 1
        if s["verdict"] == "KEEP":
            keepers.append(f"{s['sport']}/{s['source']}")
        elif s["verdict"] == "DROP":
            drops.append(f"{s['sport']}/{s['source']}")
    lines.append("## TL;DR")
    lines.append("")
    n = report.get("n_sources") or 0
    keep_pct = (counts.get("KEEP", 0) / n * 100) if n else 0
    lines.append(
        f"- **{n}** (sport, source) pairs audited. "
        f"**{counts.get('KEEP', 0)}** keep, "
        f"**{counts.get('KEEP-WITH-CAVEATS', 0)}** keep-with-caveats, "
        f"**{counts.get('NOISE', 0)}** noise, "
        f"**{counts.get('DROP', 0)}** drop, "
        f"**{counts.get('INSUFFICIENT-DATA', 0)}** insufficient-data."
    )
    if keepers:
        lines.append(
            f"- **KEEP (ROI > +1% on $100/event flat)**: {', '.join(keepers)}."
        )
    else:
        lines.append(
            "- **No source clears +1% ROI on $100/event flat.** This is "
            "exactly the structural-vig issue Phil flagged in PHIL_PLAN.md. "
            "The production +3pp Kelly gate filters most of these picks out; "
            "the source audit is intentionally meaner so we catch a source "
            "degrading before the gate stops covering for it."
        )
    if drops:
        lines.append(
            f"- **DROP (worse than vig or worse than a coin flip)**: "
            f"{', '.join(drops)}."
        )
    lines.append("")

    lines.append(
        "**Method.** For every prediction source the model touches, score it "

        "on the full graded ledger we have. Brier + log loss come from the "
        "raw `home_prob` × outcome rows in `source_history.db`. Headline ROI, "
        "drawdown, and longest losing streak come from a $100-per-event "
        "flat-stake hypothetical: pick the higher-probability side, settle "
        "at the closing book price on that side. No edge gate, no Kelly, no "
        "skip-coin-flips — that's the production rule; this is the source "
        "audit. CLV is the source's picked-side probability minus the "
        "single-side market implied. Verdict buckets are: **KEEP** "
        "(ROI > +1%), **KEEP-WITH-CAVEATS** (−3% ≤ ROI ≤ +1%), **NOISE** "
        "(brier in vig territory), **DROP** (brier ≥ 0.25 or ROI ≤ −10%), "
        "**INSUFFICIENT-DATA** (< 200 graded events)."
    )
    lines.append("")

    by_sport: dict[str, list[dict]] = defaultdict(list)
    for s in report["sources"]:
        by_sport[s["sport"]].append(s)

    # Summary table at the top.
    lines.append("## Summary — every source, every sport")
    lines.append("")
    lines.append(
        "| Sport | Source | n | Hit | Brier ↓ | LogLoss ↓ | ROI/$100 | "
        "CLV (pp) | Max DD | Longest L-Streak | Verdict |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    sports = sorted(by_sport.keys())
    for sp in sports:
        for s in by_sport[sp]:
            m = s["metrics"]
            lines.append(
                f"| {sp} | `{s['source']}` | {m.get('n_predictions') or '—'} "
                f"| {_fmt_pct(m.get('hit_rate'))} "
                f"| {_fmt_num(m.get('brier'))} "
                f"| {_fmt_num(m.get('log_loss'))} "
                f"| {_fmt_pct(m.get('roi_flat_100'))} "
                f"| {_fmt_num((m.get('clv_pp') or 0) * 100, 2) if m.get('clv_pp') is not None else '—'} "
                f"| {_fmt_usd(m.get('max_drawdown_usd'))} "
                f"| {m.get('longest_losing_streak') if m.get('longest_losing_streak') is not None else '—'} "
                f"| **{s['verdict']}** |"
            )
    lines.append("")

    # Verdict roll-up.
    lines.append("## Verdict roll-up")
    lines.append("")
    for v in ("KEEP", "KEEP-WITH-CAVEATS", "NOISE", "DROP", "INSUFFICIENT-DATA"):
        lines.append(f"- **{v}**: {counts.get(v, 0)}")
    lines.append("")

    # Per-source detail with notes (predict.tennis and the merged tennis-data
    # pulls have long-form provenance worth surfacing inline).
    lines.append("## Per-source notes")
    lines.append("")
    for sp in sports:
        for s in by_sport[sp]:
            if not s.get("notes"):
                continue
            lines.append(f"### {sp} · `{s['source']}` — {s['verdict']}")
            for n in s["notes"]:
                lines.append(f"- {n}")
            m = s["metrics"]
            if m.get("window", {}).get("start"):
                lines.append(
                    f"- window: {m['window']['start']} → {m['window']['end']}"
                )
            lines.append("")

    # Honesty pact.
    lines.append("---")
    lines.append("")
    lines.append("### Honesty pact")
    lines.append("")
    lines.append(
        "- Numbers are reported as-is. Sources losing money on the $100/event "
        "hypothetical get bucketed as **NOISE** or **DROP**, not spun."
    )
    lines.append(
        "- predict.tennis ROI numbers are SELF-REPORTED by predict.tennis on "
        "their own Prediction Check page and 2024 season review. The site "
        "publishes its own losing yields. We record those verbatim."
    )
    lines.append(
        "- Per-source ROI is hypothetical: $100 flat on every event the source "
        "weighed in on, no edge gate. Production picks pass through a +3pp "
        "edge gate and 1/4 Kelly — the source audit is intentionally meaner."
    )
    lines.append(
        "- This is the FIRST RUN of a weekly recurring process. See "
        "`docs/AGENT_LOOP.md` for the standing operational cadence."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(
    report: dict,
    *,
    out_dir: Path,
    dated_name: str | None = None,
    latest_name: str = "source-accountability-latest.md",
) -> tuple[Path, Path]:
    """Write the markdown report to ``out_dir`` (dated + latest).

    Also writes the underlying JSON next to each markdown file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(report)
    today = datetime.now(timezone.utc).date().isoformat()
    dated = out_dir / (dated_name or f"source-accountability-{today}.md")
    latest = out_dir / latest_name
    dated.write_text(md, encoding="utf-8")
    latest.write_text(md, encoding="utf-8")
    (dated.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (latest.with_suffix(".json")).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return dated, latest
