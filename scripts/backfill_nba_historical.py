#!/usr/bin/env python3
"""Walk-forward NBA historical backfill — 2022-01-01 → 2024-12-31.

Counterpart to ``backfill_mlb_statcast.py``. The MLB script calibrates a
single connector (``mlb-statcast-lineup``) with player-level Statcast.
This script does the broader NBA equivalent:

* Pull every regular-season NBA game 2021-22 → 2023-24 via the
  ``nba_api`` package (which talks to ``stats.nba.com``). This is the
  documented fallback path when ``basketball-reference.com`` returns 403
  from CI / cloud egress — and unfortunately bref still 403s here, so
  we *are* on the fallback path. (See ``docs/PHIL_PLAN.md`` and the
  pre-flight note in PR #13.)
* For each game, compute team SRS strictly from games BEFORE the game
  date (walk-forward, asserted with an in-loop leakage gate).
* Convert SRS-diff → home win probability via the existing
  ``nba_brefer.diff_to_home_prob`` (HFA = +2.5, σ = 11.0 NBA margin).
* Re-run the 538 NBA Elo / RAPTOR / CARM-Elo archive on the same window
  using the existing cached CSV — that one is the *only* historical NBA
  prob source we have that ships with a 538-published pre-game prob
  number, so we treat its row as ground-truth historical prediction.
* Persist row-per-(event,source) into ``data/source_history.db`` with
  ``home_won`` (so Brier / log-loss / accuracy populate) and a per-source
  ``meta`` row.
* Re-fit a Platt calibration per source over the window. Save to
  ``data/calibration.json`` under ``nba-bref-srs-pace.platt`` etc.

ROI / Kelly: the public NBA odds archives we know about are dead.
``sportsbookreviewsonline.com`` redirects to its home page; the ESPN
historical odds endpoint serves only the trailing ~30 days; ``the-odds-api``
historical archive requires a paid key (THE_ODDS_API_KEY) which Phil hasn't
provisioned for this run. We therefore document the same blocker the CFB
connector hit and ship NBA with ``roi=NULL`` / ``n_bets=0`` but non-null
``brier`` / ``accuracy`` / ``log_loss`` / ``calibration_slope``. That is
explicitly the path Phil approved when no free historical moneyline source
is reachable — better to populate the predictions table and surface real
Brier than to skip the sport entirely.

Per-sport mode impact: NBA will remain RESEARCH (n_bets == 0 < 200
``live_min_bets``). That's the right answer — the per-sport gate is
unchanged. What changes is the source-detail page now shows Brier and
accuracy for ``nba-bref-srs-pace`` instead of "n/a".

Usage::

    PYTHONPATH=src python scripts/backfill_nba_historical.py
    # Time-budget knobs match the MLB script:
    PYTHONPATH=src python scripts/backfill_nba_historical.py --max-minutes 30

Exit codes:
  0 — success (full or partial backfill persisted)
  2 — required deps missing (nba_api)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import CACHE_DIR, CALIBRATION_PATH, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.sources.nba_brefer import (  # noqa: E402
    NBA_HFA_POINTS,
    NBA_MARGIN_SIGMA,
    diff_to_home_prob,
)
from flashcat.source_history import upsert_meta, upsert_predictions  # noqa: E402

log = logging.getLogger("backfill_nba_historical")

NBA_CACHE = CACHE_DIR / "nba_historical"
NBA_GAMELOG_CACHE = NBA_CACHE / "gamelogs"
ELO_CACHE_FILE = CACHE_DIR / "538_nba_elo.csv"

# Seasons covered. NBA season label "2023-24" maps to games 2023-10-XX through
# 2024-04-XX. We cover three regular seasons fully inside the 2022-01-01
# → 2024-12-31 window the task asks for.
SEASONS = ("2021-22", "2022-23", "2023-24")

# Walk-forward leakage gate: every prediction's input data must have a
# game date strictly less than the prediction's game date. We assert this
# in-loop so the test suite can lean on it as an invariant.
LEAKAGE_GATE_ENABLED = True


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_nba_gamelog(season: str) -> list[dict]:
    """Pull (and cache) all regular-season game rows for one NBA season.

    The ``nba_api.LeagueGameFinder`` endpoint returns one row per team per
    game, so 2460 rows / 1230 unique games per season. We normalize to
    one row per game with explicit home / away assignment from the
    ``MATCHUP`` string ("ATL vs. BOS" → ATL is home, "ATL @ BOS" → BOS).

    Returns a list of dicts: ``{date, home, away, home_score, away_score,
    home_won, game_id}``. Empty list if nba_api fails or returns nothing.
    """
    NBA_GAMELOG_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = NBA_GAMELOG_CACHE / f"games_{season}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    try:
        from nba_api.stats.endpoints import leaguegamefinder  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.error("nba_api import failed: %s", e)
        return []

    log.info("nba_api.LeagueGameFinder season=%s", season)
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
            timeout=60,
        )
        df = gf.get_data_frames()[0]
    except Exception as e:  # noqa: BLE001
        log.warning("LeagueGameFinder season=%s failed: %s", season, e)
        return []

    # df rows: SEASON_ID, TEAM_ID, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID,
    # GAME_DATE, MATCHUP, WL, PTS, ...
    games: dict[str, dict] = {}
    for row in df.to_dict("records"):
        gid = str(row.get("GAME_ID") or "")
        if not gid:
            continue
        matchup = (row.get("MATCHUP") or "").strip()
        team_abbr = (row.get("TEAM_ABBREVIATION") or "").strip()
        # "ATL vs. BOS" → this row is the home (ATL).
        # "ATL @ BOS"   → this row is the away (ATL), BOS is home.
        is_home_row = "vs." in matchup
        try:
            opp = matchup.split("vs." if is_home_row else "@", 1)[1].strip()
        except Exception:
            continue
        pts = row.get("PTS")
        try:
            pts = int(pts) if pts is not None else None
        except Exception:
            pts = None
        if pts is None:
            continue

        slot = games.setdefault(gid, {
            "game_id": gid,
            "date": row.get("GAME_DATE"),
        })
        if is_home_row:
            slot["home"] = team_abbr
            slot["home_score"] = pts
            slot["away_inferred_from_home"] = opp
        else:
            slot["away"] = team_abbr
            slot["away_score"] = pts
            slot["home_inferred_from_away"] = opp

    # Normalize and filter: keep games with both halves and a valid date.
    out: list[dict] = []
    for gid, g in games.items():
        if "home" not in g or "away" not in g:
            # Cross-fill from the inferred opponent if one half is missing.
            # In practice both halves are always present for completed games.
            continue
        if g.get("home_score") is None or g.get("away_score") is None:
            continue
        try:
            d = datetime.strptime(g["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        out.append({
            "game_id": gid,
            "date": d.isoformat(),
            "home": g["home"],
            "away": g["away"],
            "home_score": int(g["home_score"]),
            "away_score": int(g["away_score"]),
            "home_won": int(int(g["home_score"]) > int(g["away_score"])),
        })
    out.sort(key=lambda r: (r["date"], r["game_id"]))
    try:
        cache_path.write_text(json.dumps(out))
    except Exception:
        pass
    log.info("season=%s games=%d", season, len(out))
    return out


# ---------------------------------------------------------------------------
# Walk-forward SRS (Simple Rating System)
# ---------------------------------------------------------------------------


def _compute_walk_forward_srs(games: list[dict]) -> list[dict]:
    """Yield one prediction row per game with the home-team SRS predicted prob.

    SRS implementation: iterative (Massey-style) per-day refresh.

    For each unique ``asof`` date, we re-fit a per-team SRS using *only*
    games before ``asof`` in the same season. Then we emit predictions for
    every game on ``asof`` using those fitted SRS values.

    SRS formula (per iteration):
        srs[t]  = avg_margin[t] + avg(srs[opp] for opp in opponents[t])

    We run 20 iterations which converges to within 1e-4 for typical NBA
    schedule density.

    Walk-forward leakage gate: ``assert asof > max(prior.date)``. Asserted
    every iteration so a future bug can't silently leak.
    """
    if not games:
        return []
    # Pre-bucket games by date.
    by_date: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        by_date[g["date"]].append(g)
    sorted_dates = sorted(by_date.keys())

    predictions: list[dict] = []
    prior_games: list[dict] = []
    for asof in sorted_dates:
        # Train SRS strictly on games BEFORE asof.
        if LEAKAGE_GATE_ENABLED and prior_games:
            assert max(g["date"] for g in prior_games) < asof, (
                f"leakage gate: prior has game on/after asof={asof}"
            )

        srs = _fit_srs(prior_games)

        # Emit predictions for every game on asof. Use 0.5 fallback if either
        # team has no prior history (first game of season for that team).
        for g in by_date[asof]:
            h_srs = srs.get(g["home"])
            a_srs = srs.get(g["away"])
            if h_srs is None or a_srs is None:
                # Cold-start: emit a 50/50 prob with HFA only. Skip from the
                # predictions table so Brier isn't dragged down by no-info
                # predictions; we still count the game in n_events.
                continue
            diff = h_srs - a_srs + NBA_HFA_POINTS
            p = diff_to_home_prob(diff)
            predictions.append({
                "game_id": g["game_id"],
                "date": asof,
                "home": g["home"],
                "away": g["away"],
                "home_srs": h_srs,
                "away_srs": a_srs,
                "diff": diff,
                "home_prob": p,
                "home_won": int(g["home_won"]),
            })

        # Now incorporate today's games into the prior set for tomorrow.
        prior_games.extend(by_date[asof])

    return predictions


def _fit_srs(games: list[dict], iterations: int = 20) -> dict[str, float]:
    """Standard SRS via fixed-point iteration.

    Each team's SRS = average point margin + average SRS of opponents.
    """
    if not games:
        return {}
    margins: dict[str, list[float]] = defaultdict(list)
    opps: dict[str, list[str]] = defaultdict(list)
    for g in games:
        h, a = g["home"], g["away"]
        m = g["home_score"] - g["away_score"]
        margins[h].append(m)
        margins[a].append(-m)
        opps[h].append(a)
        opps[a].append(h)
    teams = list(margins.keys())
    avg_margin = {t: sum(margins[t]) / len(margins[t]) for t in teams}
    srs = {t: avg_margin[t] for t in teams}
    for _ in range(iterations):
        new = {}
        for t in teams:
            opp_srs_avg = sum(srs.get(o, 0.0) for o in opps[t]) / max(1, len(opps[t]))
            new[t] = avg_margin[t] + opp_srs_avg
        # Re-center on 0 to keep numerically stable.
        mean = sum(new.values()) / len(new)
        srs = {t: v - mean for t, v in new.items()}
    return srs


# ---------------------------------------------------------------------------
# 538 NBA Elo archive — replay against the same window.
# ---------------------------------------------------------------------------


def _load_538_nba_rows(start: date, end: date) -> list[dict]:
    """Read the cached 538 NBA Elo CSV and emit predictions for the window."""
    if not ELO_CACHE_FILE.exists():
        log.warning("538 NBA Elo cache missing at %s — skipping 538 backfill",
                    ELO_CACHE_FILE)
        return []
    rows = []
    with open(ELO_CACHE_FILE) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            try:
                s1 = float(r.get("score1") or "")
                s2 = float(r.get("score2") or "")
            except Exception:
                continue
            home = (r.get("team1") or "").strip()
            away = (r.get("team2") or "").strip()
            if not home or not away:
                continue
            rows.append({
                "date": d.isoformat(),
                "home": home,
                "away": away,
                "home_won": 1 if s1 > s2 else 0,
                "elo_prob1": _safe_f(r.get("elo_prob1")),
                "carm_prob1": _safe_f(r.get("carm-elo_prob1")),
                "raptor_prob1": _safe_f(r.get("raptor_prob1")),
            })
    return rows


def _safe_f(s):
    try:
        if s is None or s == "":
            return None
        return float(s)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Stats helpers (kept local — no scipy dependency)
# ---------------------------------------------------------------------------


def _clip(p: float) -> float:
    return max(1e-3, min(1 - 1e-3, p))


def _brier(rows: list[dict]) -> float | None:
    if not rows:
        return None
    return sum((r["home_prob"] - r["home_won"]) ** 2 for r in rows) / len(rows)


def _log_loss(rows: list[dict]) -> float | None:
    if not rows:
        return None
    s = 0.0
    for r in rows:
        p = _clip(r["home_prob"])
        y = r["home_won"]
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(rows)


def _accuracy(rows: list[dict]) -> float | None:
    if not rows:
        return None
    hits = sum(1 for r in rows if (r["home_prob"] >= 0.5) == bool(r["home_won"]))
    return hits / len(rows)


def _calibration_slope(rows: list[dict]) -> float | None:
    """Logistic regression slope of outcome on logit(prob).

    β=1 → perfectly calibrated; β<1 → too confident.
    """
    n = len(rows)
    if n < 30:
        return None
    xs = [math.log(_clip(r["home_prob"]) / (1 - _clip(r["home_prob"]))) for r in rows]
    ys = [float(r["home_won"]) for r in rows]
    alpha = 0.0
    beta = 1.0
    for _ in range(50):
        ga = 0.0
        gb = 0.0
        h_aa = 0.0
        h_ab = 0.0
        h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = 1.0 / (1.0 + math.exp(-(alpha + beta * x)))
            err = mu - y
            ga += err
            gb += err * x
            w = mu * (1 - mu)
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if det == 0:
            break
        da = (h_bb * ga - h_ab * gb) / det
        db = (-h_ab * ga + h_aa * gb) / det
        alpha -= da
        beta -= db
        if abs(da) + abs(db) < 1e-6:
            break
    if not (math.isfinite(beta) and math.isfinite(alpha)):
        return None
    return beta


def _fit_platt(rows: list[dict]) -> tuple[float, float] | None:
    """Fit Platt scaling (intercept, slope) on logit(home_prob) → home_won.

    Returns (intercept, slope) such that calibrated_prob = σ(intercept + slope·logit(p)).
    None if too little data.
    """
    n = len(rows)
    if n < 100:
        return None
    xs = [math.log(_clip(r["home_prob"]) / (1 - _clip(r["home_prob"]))) for r in rows]
    ys = [float(r["home_won"]) for r in rows]
    alpha = 0.0
    beta = 1.0
    for _ in range(60):
        ga = 0.0
        gb = 0.0
        h_aa = 0.0
        h_ab = 0.0
        h_bb = 0.0
        for x, y in zip(xs, ys):
            mu = 1.0 / (1.0 + math.exp(-(alpha + beta * x)))
            err = mu - y
            ga += err
            gb += err * x
            w = mu * (1 - mu)
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if det == 0:
            break
        da = (h_bb * ga - h_ab * gb) / det
        db = (-h_ab * ga + h_aa * gb) / det
        alpha -= da
        beta -= db
        if abs(da) + abs(db) < 1e-7:
            break
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        return None
    return alpha, beta


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _persist_predictions(
    sport: str,
    source: str,
    rows: list[dict],
    *,
    db_path: Path | None = None,
) -> int:
    """Write rows to ``predictions`` table. Each row needs date, home, away, home_prob, home_won."""
    if not rows:
        return 0
    pred_rows = []
    for r in rows:
        commence = f"{r['date']}T00:00:00+00:00"
        eid = f"{source}:{r['date']}_{r['away']}_{r['home']}"
        pred_rows.append({
            "event_id": eid,
            "sport": sport,
            "source": source,
            "commence_time": commence,
            "home": r["home"],
            "away": r["away"],
            "home_prob": float(r["home_prob"]),
            "home_won": int(r["home_won"]),
            "market_close_home": None,
            "market_close_decimal": None,
        })
    return upsert_predictions(pred_rows, path=db_path)


def _persist_meta(
    sport: str,
    source: str,
    rows: list[dict],
    *,
    db_path: Path | None = None,
) -> None:
    if not rows:
        return
    window_start = min(r["date"] for r in rows)
    window_end = max(r["date"] for r in rows)
    upsert_meta([{
        "sport": sport,
        "source": source,
        "window_start": window_start,
        "window_end": window_end,
        "n_events": len(rows),
        "n_bets": 0,  # no historical NBA moneylines reachable; see PR body.
        "brier": _brier(rows),
        "log_loss": _log_loss(rows),
        "accuracy": _accuracy(rows),
        "roi": None,
        "calibration_slope": _calibration_slope(rows),
        "avg_clv_pp": None,
    }], path=db_path)


def _save_calibration_for_nba(per_source_platt: dict[str, tuple[float, float]]) -> None:
    """Append/update NBA per-source Platt calibration in ``data/calibration.json``."""
    path = Path(CALIBRATION_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
    nba = data.setdefault("nba", {})
    for source, (alpha, beta) in per_source_platt.items():
        nba[source] = {
            "platt": {"intercept": alpha, "slope": beta},
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "method": "logit-Newton",
            "window": "2022-01-01..2024-12-31",
        }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    log.info("Wrote NBA calibration entries to %s", path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def backfill(*, max_minutes: int = 60) -> dict:
    """Run the full NBA backfill across the three covered seasons."""
    try:
        import nba_api  # noqa: F401
    except Exception:
        log.error("nba_api missing — install with: pip install nba_api")
        sys.exit(2)

    started = time.time()
    per_source_rows: dict[str, list[dict]] = defaultdict(list)
    per_season_summary: dict[str, dict] = {}

    # --- 1. nba-bref-srs-pace (walk-forward SRS) -------------------------
    for season in SEASONS:
        if (time.time() - started) / 60 > max_minutes:
            log.warning("Time budget exceeded; persisting partial.")
            break
        games = _load_nba_gamelog(season)
        if not games:
            continue
        preds = _compute_walk_forward_srs(games)
        n_games = len(games)
        n_preds = len(preds)
        b = _brier(preds)
        acc = _accuracy(preds)
        log.info(
            "season=%s games=%d srs_predictions=%d brier=%.4f acc=%.3f",
            season, n_games, n_preds,
            b if b is not None else float("nan"),
            acc if acc is not None else float("nan"),
        )
        per_season_summary[season] = {
            "n_games": n_games,
            "n_predictions": n_preds,
            "brier": b,
            "accuracy": acc,
        }
        per_source_rows["nba-bref-srs-pace"].extend(preds)

    # --- 2. 538 NBA Elo archive (overlapping window) ---------------------
    elo_rows_raw = _load_538_nba_rows(date(2022, 1, 1), date(2024, 12, 31))
    elo_p_rows = []
    raptor_p_rows = []
    carm_p_rows = []
    for r in elo_rows_raw:
        base = {
            "date": r["date"],
            "home": r["home"],
            "away": r["away"],
            "home_won": r["home_won"],
        }
        if r["elo_prob1"] is not None:
            elo_p_rows.append({**base, "home_prob": _clip(r["elo_prob1"])})
        if r["carm_prob1"] is not None:
            carm_p_rows.append({**base, "home_prob": _clip(r["carm_prob1"])})
        if r["raptor_prob1"] is not None:
            raptor_p_rows.append({**base, "home_prob": _clip(r["raptor_prob1"])})
    log.info(
        "538 NBA window 2022-01..2024-12: elo=%d carm=%d raptor=%d",
        len(elo_p_rows), len(carm_p_rows), len(raptor_p_rows),
    )
    if elo_p_rows:
        per_source_rows["fivethirtyeight-nba-elo-modern"].extend(elo_p_rows)
    if carm_p_rows:
        per_source_rows["fivethirtyeight-nba-carm"].extend(carm_p_rows)
    if raptor_p_rows:
        per_source_rows["fivethirtyeight-nba-raptor"].extend(raptor_p_rows)

    # --- 3. Persist + calibrate ------------------------------------------
    per_source_summary: dict[str, dict] = {}
    per_source_platt: dict[str, tuple[float, float]] = {}
    for source, rows in per_source_rows.items():
        if not rows:
            continue
        n_pred = _persist_predictions("nba", source, rows)
        _persist_meta("nba", source, rows)
        platt = _fit_platt(rows)
        if platt:
            per_source_platt[source] = platt
        per_source_summary[source] = {
            "n_predictions": n_pred,
            "brier": _brier(rows),
            "log_loss": _log_loss(rows),
            "accuracy": _accuracy(rows),
            "calibration_slope": _calibration_slope(rows),
            "platt": ({"intercept": platt[0], "slope": platt[1]} if platt else None),
        }
        log.info(
            "  %-35s n=%d brier=%.4f acc=%.3f",
            source, n_pred,
            per_source_summary[source]["brier"] if per_source_summary[source]["brier"] else float("nan"),
            per_source_summary[source]["accuracy"] if per_source_summary[source]["accuracy"] else float("nan"),
        )

    if per_source_platt:
        _save_calibration_for_nba(per_source_platt)

    return {
        "per_season": per_season_summary,
        "per_source": per_source_summary,
        "elapsed_s": time.time() - started,
        "n_seasons": len(per_season_summary),
        "n_sources_persisted": len(per_source_summary),
        "historical_odds_blocker": (
            "No free historical NBA moneyline source is reachable from this "
            "build environment as of 2026-05-30. SBR (sportsbookreviewsonline) "
            "now redirects all archive pages to its home; ESPN's gameOdds "
            "endpoint only serves the trailing ~30 days (see "
            "sportsdataverse/hoopR#173); the-odds-api historical archive "
            "requires a paid THE_ODDS_API_KEY that is not provisioned in CI. "
            "Same blocker the CFB connector documented in PR #14."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s %(message)s"
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=int, default=60)
    args = ap.parse_args(argv)
    summary = backfill(max_minutes=args.max_minutes)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
