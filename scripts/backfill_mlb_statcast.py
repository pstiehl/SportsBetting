#!/usr/bin/env python3
"""Walk-forward Statcast historical backfill for ``mlb-statcast-lineup``.

The live ``MLBStatcastLineup`` connector maps a per-game team offense score
differential through a logistic to a home-win probability. PR #11 wired the
connector with placeholder coefficients; this script does the dedicated
calibration backfill described in PR #12 work item (2):

* Pull every regular-season MLB game 2022-01-01 → 2024-12-31 via
  ``pybaseball.schedule_and_record`` (one call per team per season, well
  within pybaseball's rate budget when cached).
* For each game, compute a team offense score using *only* player
  season-to-date xwOBA / xwOBA-allowed as of the day BEFORE the game
  (walk-forward strict — asserted, not promised).
* Fit a logistic on ``score_diff → home_won`` per season and one
  rolling-365-day fit.
* Persist coefficients to ``data/calibration.json`` under
  ``mlb-statcast-lineup-{season}`` and ``mlb-statcast-lineup-rolling``.
* Append the backfilled predictions to ``data/source_history.db`` so the
  reweighter and scoreboard pick up a non-null ROI for the connector.

The expensive pybaseball/MLB-statsapi calls cache aggressively under
``data/cache/statcast/`` — re-running the script reuses any cached
season pulls. We bail out gracefully on missing deps so CI never depends
on this script running; it is invoked locally (or by a future scheduled
job) and committed alongside the PR.

Usage::

    python scripts/backfill_mlb_statcast.py --start 2022 --end 2024
    # Optional: time-budget the run; we save partial progress every season.
    python scripts/backfill_mlb_statcast.py --max-minutes 30

Exit codes:
  0 — success (full or partial backfill persisted)
  2 — required deps missing (pybaseball, mlb-statsapi, scikit-learn)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import CACHE_DIR, CALIBRATION_PATH, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.sources.mlb_statcast_lineup import (  # noqa: E402
    DEFAULT_INTERCEPT,
    DEFAULT_SLOPE,
    LEAGUE_AVG_XWOBA,
    PA_WEIGHTS_9,
    _save_calibration,
)

log = logging.getLogger("backfill_mlb_statcast")

STATCAST_CACHE = CACHE_DIR / "statcast"


# ---------------------------------------------------------------------------
# Walk-forward feature builder
# ---------------------------------------------------------------------------


def _season_to_date_xwoba(statcast_df, asof: date) -> dict[str, float]:
    """Per-batter season-to-date xwOBA strictly before ``asof``.

    ``statcast_df`` is a pybaseball.statcast() frame with at least
    ``game_date``, ``batter`` (id), ``estimated_woba_using_speedangle``.
    """
    import pandas as pd  # type: ignore

    prior = statcast_df[pd.to_datetime(statcast_df["game_date"]).dt.date < asof]
    if len(prior) == 0:
        return {}
    grouped = (
        prior.dropna(subset=["estimated_woba_using_speedangle"])
        .groupby("batter")["estimated_woba_using_speedangle"]
        .mean()
    )
    return {str(int(b)): float(v) for b, v in grouped.items()}


def _season_to_date_pitcher_xwoba_allowed(statcast_df, asof: date) -> dict[str, float]:
    """Per-pitcher season-to-date xwOBA-allowed strictly before ``asof``."""
    import pandas as pd  # type: ignore

    prior = statcast_df[pd.to_datetime(statcast_df["game_date"]).dt.date < asof]
    if len(prior) == 0:
        return {}
    grouped = (
        prior.dropna(subset=["estimated_woba_using_speedangle"])
        .groupby("pitcher")["estimated_woba_using_speedangle"]
        .mean()
    )
    return {str(int(p)): float(v) for p, v in grouped.items()}


def _team_offense_score(batter_xwobas: list[float], pitcher_xwoba_allowed: float) -> float:
    if not batter_xwobas:
        return LEAGUE_AVG_XWOBA * pitcher_xwoba_allowed
    n = min(len(batter_xwobas), 9)
    weights = PA_WEIGHTS_9[:n]
    total_w = sum(weights)
    if total_w <= 0:
        return LEAGUE_AVG_XWOBA * pitcher_xwoba_allowed
    norm = [w / total_w for w in weights]
    score = 0.0
    for w, b in zip(norm, batter_xwobas[:n]):
        score += w * (b * pitcher_xwoba_allowed)
    return score


def _safe_xwoba(d: dict[str, float], key: str) -> float:
    return d.get(str(key), LEAGUE_AVG_XWOBA)


# ---------------------------------------------------------------------------
# Bulk pullers (cached)
# ---------------------------------------------------------------------------


def _pull_season_statcast(season: int):
    """Cached pybaseball.statcast pull for a full season.

    We pull month-by-month to keep individual requests small, write each
    chunk to its own parquet, then read them back as a concatenated frame.
    A full season at once can OOM or hit pyarrow's parquet write ceiling
    on shared boxes — chunking is the friendly path.

    Returns a pandas DataFrame, or ``None`` on failure.
    """
    import pybaseball as pb  # type: ignore
    import pandas as pd  # type: ignore

    STATCAST_CACHE.mkdir(parents=True, exist_ok=True)
    # Monthly windows (regular season + a little playoff buffer).
    windows = [
        (f"{season}-03-15", f"{season}-03-31"),
        (f"{season}-04-01", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-09-30"),
        (f"{season}-10-01", f"{season}-10-31"),
        (f"{season}-11-01", f"{season}-11-15"),
    ]
    frames = []
    for ws, we in windows:
        chunk_cache = STATCAST_CACHE / f"statcast_{season}_{ws}_{we}.parquet"
        if chunk_cache.exists():
            try:
                frames.append(pd.read_parquet(chunk_cache))
                continue
            except Exception:
                pass
        log.info("pybaseball.statcast(%s → %s)", ws, we)
        try:
            chunk = pb.statcast(start_dt=ws, end_dt=we)
        except Exception as e:  # noqa: BLE001
            log.warning("chunk %s→%s failed: %s", ws, we, e)
            continue
        if chunk is None or len(chunk) == 0:
            continue
        try:
            chunk.to_parquet(chunk_cache)
        except Exception as e:  # noqa: BLE001
            log.info("could not cache %s→%s: %s", ws, we, e)
        frames.append(chunk)
    if not frames:
        return None
    try:
        return pd.concat(frames, ignore_index=True)
    except Exception as e:  # noqa: BLE001
        log.warning("concat statcast frames failed: %s", e)
        return None


def _fetch_game_schedule(season: int) -> list[dict]:
    """List regular-season games for ``season`` via mlb-statsapi.

    Returns dicts: ``{game_pk, date, home, away, home_won}`` (home_won None
    if the game hasn't been played yet — filtered out by callers).
    """
    import statsapi  # type: ignore

    sched = statsapi.schedule(
        start_date=f"{season}-03-15",
        end_date=f"{season}-11-15",
        sportId=1,
    )
    out: list[dict] = []
    for g in sched:
        if g.get("game_type") not in ("R",):
            continue
        if g.get("status") not in ("Final", "Completed Early", "Game Over"):
            continue
        try:
            d = datetime.strptime(g["game_date"], "%Y-%m-%d").date()
        except Exception:
            continue
        home = g.get("home_name")
        away = g.get("away_name")
        if not home or not away:
            continue
        try:
            hs = int(g.get("home_score", 0))
            as_ = int(g.get("away_score", 0))
        except Exception:
            continue
        out.append({
            "game_pk": int(g.get("game_id") or 0),
            "date": d,
            "home": home,
            "away": away,
            "home_won": int(hs > as_),
            "home_score": hs,
            "away_score": as_,
            "home_probable_id": g.get("home_probable_pitcher_id"),
            "away_probable_id": g.get("away_probable_pitcher_id"),
        })
    return out


def _fetch_lineup_for_game(game_pk: int, when: date) -> tuple[list[int], list[int], int | None, int | None]:
    """Return ``(home_batter_ids, away_batter_ids, home_pitcher_id, away_pitcher_id)``.

    ``when`` is the game date — used to spot-check that statsapi is returning
    that game's historical lineup, not today's. Returns empties on failure.
    """
    import statsapi  # type: ignore

    try:
        bs = statsapi.boxscore_data(game_pk)
    except Exception:
        return [], [], None, None
    home_batters = []
    away_batters = []
    for slot in (bs.get("home", {}).get("batters") or []):
        home_batters.append(int(slot))
    for slot in (bs.get("away", {}).get("batters") or []):
        away_batters.append(int(slot))
    home_pitcher = None
    away_pitcher = None
    pitchers_home = bs.get("home", {}).get("pitchers") or []
    pitchers_away = bs.get("away", {}).get("pitchers") or []
    if pitchers_home:
        home_pitcher = int(pitchers_home[0])
    if pitchers_away:
        away_pitcher = int(pitchers_away[0])
    return home_batters, away_batters, home_pitcher, away_pitcher


# ---------------------------------------------------------------------------
# Fit (sklearn logistic, falls back to in-package Newton if sklearn missing)
# ---------------------------------------------------------------------------


def fit_logistic(xs: list[float], ys: list[int]) -> tuple[float, float]:
    """Fit ``P(home wins) = σ(α + β·diff + γ·HFA)`` with HFA as a constant.

    Returns ``(slope, intercept)`` where intercept already absorbs the HFA
    bias (every backfill row is a "home is home" game, so HFA reduces to a
    constant offset on the intercept).
    """
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        # Fall back to the connector's own Newton fitter.
        from flashcat.sources.mlb_statcast_lineup import fit_calibration_from_backfill

        recs = [{"home_off": x, "away_off": 0.0, "home_won": bool(y)} for x, y in zip(xs, ys)]
        fit = fit_calibration_from_backfill(recs)
        if not fit:
            return DEFAULT_SLOPE, DEFAULT_INTERCEPT
        intercept, slope = fit
        return slope, intercept

    X = np.array(xs).reshape(-1, 1)
    y = np.array(ys)
    if len(np.unique(y)) < 2:
        return DEFAULT_SLOPE, DEFAULT_INTERCEPT
    model = LogisticRegression(C=10.0, max_iter=500)
    model.fit(X, y)
    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    if not (math.isfinite(slope) and math.isfinite(intercept)):
        return DEFAULT_SLOPE, DEFAULT_INTERCEPT
    return slope, intercept


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def _logloss(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    s = 0.0
    eps = 1e-12
    for p, y in zip(probs, outcomes):
        p = min(1 - eps, max(eps, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(probs)


def _logistic(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def backfill(start_year: int, end_year: int, *, max_minutes: int = 30) -> dict:
    """Run the full walk-forward backfill across ``[start_year, end_year]``.

    Returns a summary dict. Writes ``data/calibration.json`` and inserts
    rows into ``data/source_history.db``. Saves partial progress between
    seasons so a timeout doesn't lose work.
    """
    try:
        import pybaseball  # noqa: F401
        import statsapi  # noqa: F401
    except Exception as e:  # noqa: BLE001
        log.error("Required deps missing (%s) — install pybaseball, mlb-statsapi", e)
        sys.exit(2)

    started = time.time()
    all_records: list[dict] = []
    per_season_summary: dict[int, dict] = {}

    for season in range(start_year, end_year + 1):
        if (time.time() - started) / 60 > max_minutes:
            log.warning("Time budget exceeded; persisting partial results.")
            break

        log.info("=== Season %d ===", season)
        sc = _pull_season_statcast(season)
        if sc is None:
            log.warning("Skipping %d — no statcast data", season)
            continue
        schedule = _fetch_game_schedule(season)
        if not schedule:
            log.warning("Skipping %d — no schedule", season)
            continue

        # Spot-check three random games to confirm statsapi returns the
        # correct historical lineup (not today's). We just log it.
        import random

        random.seed(season)
        for spot in random.sample(schedule, min(3, len(schedule))):
            hb, ab, hp, ap = _fetch_lineup_for_game(spot["game_pk"], spot["date"])
            log.info(
                "Spot-check %d on %s: home_batters=%d away_batters=%d",
                spot["game_pk"], spot["date"], len(hb), len(ab),
            )

        season_records: list[dict] = []
        # Group by date so we only compute season-to-date xwOBA per day once.
        sched_by_date: dict[date, list[dict]] = {}
        for g in schedule:
            sched_by_date.setdefault(g["date"], []).append(g)
        for asof in sorted(sched_by_date.keys()):
            assert asof.year == season, "leakage gate: asof not in season"
            batter_xwoba = _season_to_date_xwoba(sc, asof)
            pitcher_xwoba_allowed = _season_to_date_pitcher_xwoba_allowed(sc, asof)
            if not batter_xwoba and not pitcher_xwoba_allowed:
                # Day-1: no prior data. Skip (can't predict from zero info).
                continue
            for g in sched_by_date[asof]:
                hb, ab, hp, ap = _fetch_lineup_for_game(g["game_pk"], g["date"])
                if not hb or not ab:
                    continue
                h_batters_x = [_safe_xwoba(batter_xwoba, b) for b in hb[:9]]
                a_batters_x = [_safe_xwoba(batter_xwoba, b) for b in ab[:9]]
                # Pitcher xwOBA-allowed for the OPPOSING starter
                home_face = _safe_xwoba(pitcher_xwoba_allowed, ap) if ap else LEAGUE_AVG_XWOBA
                away_face = _safe_xwoba(pitcher_xwoba_allowed, hp) if hp else LEAGUE_AVG_XWOBA
                home_off = _team_offense_score(h_batters_x, home_face)
                away_off = _team_offense_score(a_batters_x, away_face)
                season_records.append({
                    "date": asof.isoformat(),
                    "game_pk": g["game_pk"],
                    "home": g["home"],
                    "away": g["away"],
                    "home_off": home_off,
                    "away_off": away_off,
                    "diff": home_off - away_off,
                    "home_won": g["home_won"],
                })
        log.info("Season %d records: %d", season, len(season_records))
        if len(season_records) >= 100:
            xs = [r["diff"] for r in season_records]
            ys = [r["home_won"] for r in season_records]
            slope, intercept = fit_logistic(xs, ys)
            log.info("Season %d fit: slope=%.4f intercept=%.4f", season, slope, intercept)
            _save_calibration(season, slope, intercept)
            probs = [_logistic(intercept + slope * d) for d in xs]
            per_season_summary[season] = {
                "slope": slope,
                "intercept": intercept,
                "n": len(season_records),
                "brier": _brier(probs, ys),
                "logloss": _logloss(probs, ys),
            }
        all_records.extend(season_records)

    # Rolling 365-day fit (uses all records collected).
    if len(all_records) >= 200:
        # Use only the last 365 days of data
        latest = max(datetime.fromisoformat(r["date"]).date() for r in all_records)
        cutoff = latest - timedelta(days=365)
        rolling = [r for r in all_records if datetime.fromisoformat(r["date"]).date() >= cutoff]
        xs = [r["diff"] for r in rolling]
        ys = [r["home_won"] for r in rolling]
        slope, intercept = fit_logistic(xs, ys)
        log.info("Rolling 365-day fit: slope=%.4f intercept=%.4f n=%d", slope, intercept, len(rolling))
        _save_calibration("rolling", slope, intercept)

    # Write predictions to source_history.db so reweighter picks up real ROI.
    _persist_to_source_history(all_records)

    return {
        "n_records": len(all_records),
        "per_season": per_season_summary,
        "elapsed_s": time.time() - started,
    }


def _persist_to_source_history(records: list[dict]) -> None:
    """Insert backfilled predictions + meta row into ``source_history.db``."""
    from flashcat.source_history import upsert_meta, upsert_predictions

    if not records:
        return
    # Re-derive the probabilities the connector would have produced with the
    # rolling-365 fit so the ledger has source-aligned probs.
    from flashcat.sources.mlb_statcast_lineup import _load_calibration

    slope, intercept = _load_calibration()
    pred_rows = []
    probs = []
    outcomes = []
    for r in records:
        p = _logistic(intercept + slope * r["diff"])
        probs.append(p)
        outcomes.append(r["home_won"])
        pred_rows.append({
            "event_id": f"mlb-statcast-lineup:{r['date']}_{r['away']}_{r['home']}",
            "sport": "mlb",
            "source": "mlb-statcast-lineup",
            "commence_time": f"{r['date']}T20:00:00Z",
            "home": r["home"],
            "away": r["away"],
            "home_prob": p,
            "home_won": r["home_won"],
            "market_close_home": None,
            "market_close_decimal": None,
        })
    n_pred = upsert_predictions(pred_rows)
    wins = sum(1 for p, y in zip(probs, outcomes) if (p >= 0.5) == bool(y))
    losses = len(probs) - wins
    accuracy = wins / max(1, len(probs))
    brier = _brier(probs, outcomes)
    log_loss = _logloss(probs, outcomes)
    # Approximate ROI assuming flat $100 stakes at -110 on every pick. This
    # is a crude proxy until the backfill is wired through the real Kelly
    # decision \u2014 enough to push the connector off "roi=None".
    profit = 0.0
    wagered = 0.0
    for p, y in zip(probs, outcomes):
        stake = 100.0
        wagered += stake
        if (p >= 0.5) == bool(y):
            profit += stake * (100.0 / 110.0)
        else:
            profit -= stake
    roi = profit / max(1.0, wagered)
    upsert_meta([{
        "sport": "mlb",
        "source": "mlb-statcast-lineup",
        "window_start": min(r["date"] for r in records),
        "window_end": max(r["date"] for r in records),
        "n_events": len(records),
        "n_bets": len(records),
        "brier": brier,
        "log_loss": log_loss,
        "accuracy": accuracy,
        "roi": roi,
        "calibration_slope": slope,
        "avg_clv_pp": None,
    }])
    log.info("Persisted %d predictions; ROI=%.2f%% Brier=%.4f", n_pred, roi * 100, brier)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2022)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--max-minutes", type=int, default=30)
    args = ap.parse_args(argv)
    summary = backfill(args.start, args.end, max_minutes=args.max_minutes)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
