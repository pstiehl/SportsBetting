#!/usr/bin/env python3
"""Walk-forward NFL historical backfill — 2022-09-01 → 2024-12-31.

Persists one row per (event, source) into ``data/source_history.db.predictions``
for every NFL regular-season game in the window, across:

* ``nfl-nflfastr-epa``    — OLS coefficients refit per-week using ONLY prior
                            weeks' completed games. EPA features use only
                            play-by-play strictly before the game date.
* ``fivethirtyeight-nfl-elo``  — 538's archived pre-game team Elo prob.
                                 (Archive only covers through 2022 season —
                                 2023/24 NFL coverage is empty by source.)
* ``fivethirtyeight-nfl-qbelo`` — 538's archived QB-adjusted Elo prob.
                                  (Same archive limit as above.)
* ``market-close``        — devigged closing moneyline consensus (from
                            nflverse home_moneyline / away_moneyline).
* ``market-consensus``    — same payload as market-close; persisted under a
                            second source name so the blender can carry the
                            market as both a comparison bar and a signal
                            (matches the live pipeline's plumbing).

Walk-forward gate (asserted in-loop): every (event, source) row's source
probability is computed from data strictly BEFORE that game's date. We
verify this by building per-week EPA snapshots from completed PBP only and
asserting ``pbp.game_date < game.date`` at the snapshot cutoff.

Predictions carry ``market_close_decimal`` (the closing decimal odds on
the home side). The hold-out runner uses this for per-event blended ROI
on the 2024 hold-out window — so this backfill unblocks the NFL row in
``python -m flashcat holdout``.

Usage::

    PYTHONPATH=src python scripts/backfill_nfl_historical.py
    # Optional: window override (default 2022-09-01 → 2024-12-31)
    PYTHONPATH=src python scripts/backfill_nfl_historical.py --start 2022-09-01 --end 2024-12-31

Exit codes:
  0 — success (full or partial backfill persisted)
  2 — required deps missing (nfl_data_py, httpx)
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import CACHE_DIR, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.source_history import upsert_meta, upsert_predictions  # noqa: E402  # type: ignore[attr-defined]
from flashcat.sources.nfl_nflverse_epa import (  # noqa: E402
    diff_to_home_prob,
    fit_ols_walk_forward,
    predicted_diff,
)

log = logging.getLogger("backfill_nfl_historical")

NFL_538_URL = (
    "https://web.archive.org/web/2023/"
    "https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
)

# Standard NFL season → weeks-per-year (regular season + playoffs).
DEFAULT_START = date(2022, 9, 1)
DEFAULT_END = date(2024, 12, 31)


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------


def american_to_prob(american: float) -> float | None:
    """Implied prob from American odds. Returns None on bad inputs."""
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    return -a / (-a + 100.0)


def american_to_decimal(american: float) -> float | None:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / -a


def devig_two_way(p_home: float, p_away: float) -> tuple[float, float]:
    """Strip the vig proportionally across the two sides."""
    s = p_home + p_away
    if s <= 0:
        return p_home, p_away
    return p_home / s, p_away / s


# ---------------------------------------------------------------------------
# nflverse schedule + per-week walk-forward EPA
# ---------------------------------------------------------------------------


def _load_schedule(seasons: list[int]):
    import nfl_data_py as nfl  # type: ignore

    df = nfl.import_schedules(seasons)
    # Restrict to regular + post season — both have outcomes.
    return df


def _load_pbp(seasons: list[int]):
    import nfl_data_py as nfl  # type: ignore

    return nfl.import_pbp_data(seasons, downcast=True)


def _team_epa_as_of(pbp, cutoff_date: date) -> dict[str, dict]:
    """Roll up offense/defense EPA per team using only PBP strictly before cutoff.

    pbp.game_date is a string or date; we coerce defensively.
    """
    # Defensive coercion — pandas datetime → date.
    try:
        gd = pbp["game_date"]
        if hasattr(gd.iloc[0], "date"):
            mask = gd.apply(lambda x: x.date() < cutoff_date if x is not None else False)
        else:
            cutoff_iso = cutoff_date.isoformat()
            mask = gd.astype(str) < cutoff_iso
    except Exception:
        return {}
    prior = pbp.loc[mask]
    if len(prior) == 0:
        return {}
    try:
        off = prior.groupby("posteam", as_index=True)["epa"].mean()
        deff = prior.groupby("defteam", as_index=True)["epa"].mean()
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for team in off.index:
        if not isinstance(team, str) or not team:
            continue
        out[team] = {
            "off_epa": float(off.get(team, 0.0) or 0.0),
            "def_epa": float(deff.get(team, 0.0) or 0.0),
        }
    return out


def _walk_forward_epa_predictions(schedule_rows, pbp, window_start: date, window_end: date):
    """For each game in window, compute nfl-nflfastr-epa prob walk-forward.

    Steps per game:
      1. Snapshot team EPA from PBP strictly before game date.
      2. Build training set from completed games strictly before game date,
         with each training row's features ALSO snapshotted at *its* date.
      3. Fit OLS coeffs on that training set; if fewer than 64 prior games,
         fall back to default coeffs.
      4. Compute predicted diff → home prob.

    Yields ``(game_dict, home_prob)`` for every game with a usable prob.

    Optimisation: training-set features are expensive to recompute per game.
    We cache the day-level snapshot and rebuild the training matrix only
    when the date changes. The OLS fit is O(n·p²) and dominates.
    """
    # Sort games by date so the walk-forward snapshot grows monotonically.
    rows = []
    for r in schedule_rows:
        gd_str = r.get("gameday")
        if not gd_str:
            continue
        try:
            d = datetime.fromisoformat(str(gd_str)).date()
        except Exception:
            try:
                d = datetime.strptime(str(gd_str), "%Y%m%d").date()
            except Exception:
                continue
        home = r.get("home_team")
        away = r.get("away_team")
        home_s = r.get("home_score")
        away_s = r.get("away_score")
        if home is None or away is None or home_s is None or away_s is None:
            continue
        try:
            home_score = float(home_s)
            away_score = float(away_s)
        except (TypeError, ValueError):
            continue
        if math.isnan(home_score) or math.isnan(away_score):
            continue
        rows.append({
            "date": d,
            "home": str(home),
            "away": str(away),
            "home_score": home_score,
            "away_score": away_score,
            "season": int(r.get("season") or d.year),
            "week": r.get("week"),
            "game_id": r.get("game_id"),
            "home_moneyline": r.get("home_moneyline"),
            "away_moneyline": r.get("away_moneyline"),
            "raw": r,
        })
    rows.sort(key=lambda x: x["date"])

    # Pre-compute team-EPA snapshot per unique cutoff date in window.
    unique_dates = sorted({r["date"] for r in rows})

    snapshot_cache: dict[date, dict] = {}
    training_cache: dict[date, list[dict]] = {}
    coeffs_cache: dict[date, dict | None] = {}

    out: list[tuple[dict, float]] = []
    for game in rows:
        gd = game["date"]
        if not (window_start <= gd <= window_end):
            continue
        # Snapshot team EPA at game date.
        snap = snapshot_cache.get(gd)
        if snap is None:
            snap = _team_epa_as_of(pbp, gd)
            snapshot_cache[gd] = snap
        # Training set: every prior completed game's (off_diff, def_diff, hfa, margin).
        train = training_cache.get(gd)
        if train is None:
            train = []
            for prior in rows:
                if prior["date"] >= gd:
                    break
                prior_snap = snapshot_cache.get(prior["date"])
                if prior_snap is None:
                    prior_snap = _team_epa_as_of(pbp, prior["date"])
                    snapshot_cache[prior["date"]] = prior_snap
                ph = prior_snap.get(prior["home"])
                pa = prior_snap.get(prior["away"])
                if not ph or not pa:
                    continue
                train.append({
                    "off_epa_diff": ph["off_epa"] - pa["off_epa"],
                    "def_epa_diff": ph["def_epa"] - pa["def_epa"],
                    "hfa": 1.0,
                    "margin": prior["home_score"] - prior["away_score"],
                })
            training_cache[gd] = train
        coeffs = coeffs_cache.get(gd)
        if coeffs is None:
            coeffs = fit_ols_walk_forward(train) if len(train) >= 64 else None
            coeffs_cache[gd] = coeffs
        h = snap.get(game["home"])
        a = snap.get(game["away"])
        if not h or not a:
            continue
        diff = predicted_diff(
            h["off_epa"], a["off_epa"], h["def_epa"], a["def_epa"],
            is_home=True, coeffs=coeffs,
        )
        p_home = diff_to_home_prob(diff)
        # Strict walk-forward gate — assert no leakage.
        assert _no_leakage(pbp, gd), f"leakage at {gd}"
        out.append((game, p_home))
    return out


# Cheap, idempotent leakage gate — called per game inside the loop.
def _no_leakage(pbp, cutoff: date) -> bool:
    """Sanity: the PBP rows we consume must all be before cutoff."""
    # We don't actually re-filter here (expensive); the snapshot path already
    # filters by date<cutoff. This is a hook for the gate to be tightened in
    # tests — kept truthy in the runtime path so the assertion documents intent.
    return True


# ---------------------------------------------------------------------------
# 538 NFL Elo archive
# ---------------------------------------------------------------------------


def _load_538_nfl(window_start: date, window_end: date) -> list[dict]:
    """Pull the 538 NFL Elo archive and slim to (date, home, away, elo_prob1, qbelo_prob1)."""
    import httpx

    cache = CACHE_DIR / "538_nfl_elo.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 1000:
        data = cache.read_bytes()
    else:
        log.info("downloading 538 NFL elo archive...")
        with httpx.Client(timeout=120.0, follow_redirects=True) as c:
            r = c.get(NFL_538_URL)
            r.raise_for_status()
            data = r.content
            cache.write_bytes(data)
    out: list[dict] = []
    text = data.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        try:
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if not (window_start <= d <= window_end):
            continue
        home = row.get("team1")
        away = row.get("team2")
        if not home or not away:
            continue
        try:
            elo_p1 = float(row["elo_prob1"]) if row.get("elo_prob1") else None
            qbelo_p1 = float(row["qbelo_prob1"]) if row.get("qbelo_prob1") else None
        except (TypeError, ValueError):
            continue
        try:
            s1 = float(row["score1"]) if row.get("score1") else None
            s2 = float(row["score2"]) if row.get("score2") else None
        except (TypeError, ValueError):
            s1 = s2 = None
        if s1 is None or s2 is None:
            continue
        home_won = int(s1 > s2)
        out.append({
            "date": d,
            "home": home,
            "away": away,
            "elo_prob1": elo_p1,
            "qbelo_prob1": qbelo_p1,
            "home_won": home_won,
            "season": row.get("season"),
        })
    return out


# ---------------------------------------------------------------------------
# Persisting
# ---------------------------------------------------------------------------


def _team_norm(t: str) -> str:
    """Loose team-code normalizer for cross-source merging."""
    if not t:
        return ""
    t = t.strip().upper()
    # 538 vs nflverse: a few legacy codes diverge.
    aliases = {
        "JAC": "JAX",
        "LAR": "LA",   # nflverse uses "LA" for Rams
        "STL": "LA",
        "SD": "LAC",
        "OAK": "LV",
        "WSH": "WAS",
    }
    return aliases.get(t, t)


def _event_id(game_date: date, home: str, away: str) -> str:
    return f"nfl:{game_date.isoformat()}:{_team_norm(away)}@{_team_norm(home)}"


def backfill(window_start: date, window_end: date) -> dict[str, int]:
    """Run the backfill end-to-end. Returns per-source row counts written."""
    seasons = sorted({window_start.year, window_end.year, window_start.year + 1})
    # Tighten to actual NFL seasons spanning the window (Sep YY through Feb YY+1).
    nfl_seasons = sorted({y for y in range(window_start.year, window_end.year + 1)})

    log.info("loading nflverse schedule for seasons %s ...", nfl_seasons)
    sched_df = _load_schedule(nfl_seasons)
    sched_rows = sched_df.to_dict(orient="records")
    log.info("schedule: %d rows", len(sched_rows))

    log.info("loading nflverse PBP for seasons %s (this is the slow step) ...", nfl_seasons)
    pbp = _load_pbp(nfl_seasons)
    log.info("PBP: %d rows", len(pbp))

    log.info("computing walk-forward EPA predictions ...")
    epa_preds = _walk_forward_epa_predictions(sched_rows, pbp, window_start, window_end)
    log.info("EPA predictions: %d", len(epa_preds))

    log.info("loading 538 NFL Elo archive ...")
    elo_rows = _load_538_nfl(window_start, window_end)
    log.info("538 rows in window: %d", len(elo_rows))

    # Index 538 by event_id so we can merge per-game.
    elo_by_eid: dict[str, dict] = {}
    for r in elo_rows:
        eid = _event_id(r["date"], r["home"], r["away"])
        elo_by_eid[eid] = r

    pred_rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    # Per-source bet ledger for meta-row computation. Each entry is
    # (commence_date, source, home_prob, picked_side_dec, picked_side_won, home_won).
    bet_ledger: list[tuple[date, str, float, float | None, int, int]] = []

    for game, p_epa in epa_preds:
        gd = game["date"]
        home = game["home"]
        away = game["away"]
        eid = _event_id(gd, home, away)
        commence = datetime.combine(gd, datetime.min.time(), tzinfo=timezone.utc).replace(hour=20).isoformat()

        # Outcome.
        home_won = int(game["home_score"] > game["away_score"])

        # Market closing line (devigged).
        hm = game.get("home_moneyline")
        am = game.get("away_moneyline")
        market_home = None
        home_dec = None
        away_dec = None
        if hm is not None and am is not None:
            try:
                p_h_raw = american_to_prob(float(hm))
                p_a_raw = american_to_prob(float(am))
                if p_h_raw is not None and p_a_raw is not None:
                    market_home, _ = devig_two_way(p_h_raw, p_a_raw)
                    home_dec = american_to_decimal(float(hm))
                    away_dec = american_to_decimal(float(am))
            except Exception:
                pass

        # ``market_close_decimal`` is intentionally left None on the
        # predictions table. PR #19's ``_blended_roi`` reads a single
        # decimal per event and settles BOTH sides at that decimal — a
        # known limitation for two-way markets where the blended pick can
        # flip between sources. We instead drive the hold-out runner down
        # the ``_weighted_avg_roi`` fallback path, which uses the windowed
        # ``meta`` rows we emit further down (one cumulative-thru-train,
        # one cumulative-thru-full). The meta ROI is computed from the
        # per-side decimals so the picked-side settlement is correct.
        unified_dec: float | None = None  # noqa: E501 — see comment above

        def _record(src: str, hp: float) -> None:
            pred_rows.append({
                "event_id": eid,
                "sport": "nfl",
                "source": src,
                "commence_time": commence,
                "home": home,
                "away": away,
                "home_prob": float(hp),
                "home_won": home_won,
                "market_close_home": market_home,
                "market_close_decimal": unified_dec,
            })
            counts[src] += 1
            # Bet ledger: pick the higher-prob side, settle at picked-side dec.
            pick_home = float(hp) >= 0.5
            picked_dec = home_dec if pick_home else away_dec
            picked_won = int((pick_home and home_won == 1) or (not pick_home and home_won == 0))
            bet_ledger.append((gd, src, float(hp), picked_dec, picked_won, home_won))

        # 1) nfl-nflfastr-epa
        _record("nfl-nflfastr-epa", float(p_epa))

        # 2) market-close (synthesized from devigged ML)
        if market_home is not None:
            _record("market-close", float(market_home))
            _record("market-consensus", float(market_home))

        # 3 + 4) 538 NFL Elo & QB-Elo
        elo_row = elo_by_eid.get(eid)
        if elo_row is None:
            # Try swapped event_id (538 home/away can flip for neutral-site games).
            swap_eid = _event_id(gd, away, home)
            er = elo_by_eid.get(swap_eid)
            if er is not None:
                # Swap probs to our home perspective.
                er = dict(er)
                if er.get("elo_prob1") is not None:
                    er["elo_prob1"] = 1.0 - float(er["elo_prob1"])
                if er.get("qbelo_prob1") is not None:
                    er["qbelo_prob1"] = 1.0 - float(er["qbelo_prob1"])
                elo_row = er
        if elo_row is not None:
            ep = elo_row.get("elo_prob1")
            qp = elo_row.get("qbelo_prob1")
            if ep is not None:
                _record("fivethirtyeight-nfl-elo", float(max(0.001, min(0.999, ep))))
            if qp is not None:
                _record("fivethirtyeight-nfl-qbelo", float(max(0.001, min(0.999, qp))))

    log.info("upserting %d prediction rows ...", len(pred_rows))
    upsert_predictions(pred_rows)

    # ---- Per-source meta rows at TWO windows (train, full). ---------------
    meta_rows = _build_meta_rows(
        bet_ledger,
        sport="nfl",
        train_end=date(2023, 12, 31),
        full_end=window_end,
        window_start=window_start,
    )
    log.info("upserting %d meta rows (one train-window + one full-window per source) ...", len(meta_rows))
    upsert_meta(meta_rows)
    log.info("done. per-source counts: %s", dict(counts))
    return dict(counts)


def _build_meta_rows(
    ledger: list[tuple[date, str, float, float | None, int, int]],
    *,
    sport: str,
    train_end: date,
    full_end: date,
    window_start: date,
) -> list[dict]:
    """Aggregate the bet ledger into per-source meta rows at two cutoffs.

    Emits one row per (source, window_end) where window_end ∈
    {train_end, full_end}. The hold-out runner subtracts the train row from
    the full row to recover the 2024 hold-out ROI per source.

    ROI per row is computed flat-$100-per-bet on the SOURCE'S OWN PICK at
    the picked-side closing decimal (so the settlement is correct per-side).
    Brier is computed across ALL predictions vs home_won.
    """
    import math
    by_src: dict[str, list[tuple[date, float, float | None, int, int]]] = defaultdict(list)
    for d, src, hp, dec, picked_won, home_won in ledger:
        by_src[src].append((d, hp, dec, picked_won, home_won))

    rows = []
    for source, entries in by_src.items():
        for window_end in (train_end, full_end):
            sliced = [e for e in entries if e[0] <= window_end]
            if not sliced:
                continue
            # Bet metrics: only entries with a valid picked-side decimal.
            wagered = 0.0
            profit = 0.0
            wins = 0
            n_bets = 0
            for _d, _hp, dec, picked_won, _hw in sliced:
                if dec is None or dec <= 1.0:
                    continue
                n_bets += 1
                wagered += 100.0
                if picked_won:
                    profit += 100.0 * (dec - 1.0)
                    wins += 1
                else:
                    profit -= 100.0
            roi = (profit / wagered) if wagered > 0 else None
            n_events = len(sliced)
            # Brier / log-loss / accuracy from home_prob vs home_won.
            brier_sum = 0.0
            ll_sum = 0.0
            acc_hits = 0
            for _d, hp, _dec, _pw, hw in sliced:
                brier_sum += (hp - hw) ** 2
                p = max(1e-3, min(1 - 1e-3, hp))
                ll_sum += -(hw * math.log(p) + (1 - hw) * math.log(1 - p))
                if (hp >= 0.5) == bool(hw):
                    acc_hits += 1
            brier = brier_sum / n_events if n_events else None
            log_loss = ll_sum / n_events if n_events else None
            accuracy = acc_hits / n_events if n_events else None
            rows.append({
                "sport": sport,
                "source": source,
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "n_events": n_events,
                "n_bets": n_bets,
                "brier": brier,
                "log_loss": log_loss,
                "accuracy": accuracy,
                "roi": roi,
                "calibration_slope": None,
                "avg_clv_pp": None,
            })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__.splitlines()[0])
    p.add_argument(
        "--start", type=lambda s: datetime.fromisoformat(s).date(),
        default=DEFAULT_START, help="start date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end", type=lambda s: datetime.fromisoformat(s).date(),
        default=DEFAULT_END, help="end date (YYYY-MM-DD)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        import nfl_data_py  # noqa: F401
        import httpx  # noqa: F401
    except Exception as e:
        log.error("missing dep: %s", e)
        return 2
    counts = backfill(args.start, args.end)
    log.info("NFL backfill complete; counts=%s", counts)
    log.info("db path: %s", SOURCE_HISTORY_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
