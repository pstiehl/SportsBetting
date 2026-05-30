#!/usr/bin/env python3
"""Walk-forward ATP & WTA historical backfill — 2022-01-01 → 2024-12-31.

Persists one row per (event, source) into ``data/source_history.db.predictions``
for every main-tour singles match in the window, across:

* ``sackmann-atp-elo`` / ``sackmann-wta-elo``
    Surface-adjusted Elo, computed walk-forward from Sackmann's
    ``tennis_atp`` / ``tennis_wta`` GitHub repos. Each match's rating is
    snapshotted strictly BEFORE the match is played.

* ``tennis-rank-bt``
    Bradley-Terry on ATP/WTA rank points as of the match. Carried directly
    from tennis-data.co.uk's ``WPts``/``LPts`` columns (which are the
    published ATP/WTA points at match time).

* ``market-close``
    Devigged closing two-way moneyline from tennis-data.co.uk
    (Pinnacle ``PSW``/``PSL`` preferred, ``B365W``/``B365L`` fallback,
    ``AvgW``/``AvgL`` last-resort).

* ``market-consensus``
    Same payload as ``market-close``, persisted under a parallel name so
    the blender carries both (matches the live pipeline's plumbing).

Walk-forward gate: Sackmann Elo is naturally walk-forward — the
``_SackmannElo`` engine updates ratings only AFTER each match in
chronological order, so the per-match prediction it yields used only
prior-match ratings. We feed that engine matches strictly in date order
across the window and persist the per-match prediction directly.

Outcomes come from tennis-data.co.uk (winner/loser columns) — we
canonicalize the alphabetically-first player as "home" to match the
existing ``tennis-rank-bt`` source's convention so event_ids merge.

Usage::

    PYTHONPATH=src python scripts/backfill_tennis_historical.py
    # Filter to one tour:
    PYTHONPATH=src python scripts/backfill_tennis_historical.py --tour atp
    PYTHONPATH=src python scripts/backfill_tennis_historical.py --tour wta
    # Window override (default 2022-01-01 → 2024-12-31):
    PYTHONPATH=src python scripts/backfill_tennis_historical.py --start 2022-01-01 --end 2024-12-31

Exit codes:
  0 — success (full or partial backfill persisted)
  2 — required deps missing (openpyxl, httpx)
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flashcat.config import CACHE_DIR, SOURCE_HISTORY_DB_PATH  # noqa: E402
from flashcat.source_history import upsert_meta, upsert_predictions  # noqa: E402
from flashcat.sources.sackmann_elo import (  # noqa: E402
    _SackmannElo,
    _canonical_pair as _sackmann_canonical_pair,
    _normalize_name,
)
from flashcat.sources.tennis_history import (  # noqa: E402
    ATP_URL_TMPL,
    WTA_URL_TMPL,
    _canonical_pair as _td_canonical_pair,
    _rank_points_prob,
)

log = logging.getLogger("backfill_tennis_historical")

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2024, 12, 31)


# ---------------------------------------------------------------------------
# Odds + name helpers
# ---------------------------------------------------------------------------


def _devig_two_way(p_h: float, p_a: float) -> tuple[float, float]:
    s = p_h + p_a
    if s <= 0:
        return p_h, p_a
    return p_h / s, p_a / s


def _decimal_to_prob(dec: float) -> float | None:
    try:
        d = float(dec)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    return 1.0 / d


def _norm(name: str) -> str:
    """Loose name normalizer for tennis-data ↔ Sackmann merging.

    tennis-data names look like:
      - ``Sabalenka A.``        (lastname firstinitial)
      - ``O Connell C.``        (multi-word lastname; last token is initial+'.')
      - ``Van Assche L.``       (multi-word lastname)
    Sackmann names look like ``Aryna Sabalenka`` (firstname lastname),
    or ``Borna Coric``, ``Casper Ruud``, etc.

    Output: ``lastname firstinitial`` lower-cased, with all whitespace
    collapsed inside the lastname.
    """
    s = (name or "").strip()
    if not s:
        return ""
    tokens = s.split()
    # tennis-data form: last token ends with '.' (e.g. ``A.``) → it's the initial.
    if tokens and tokens[-1].endswith(".") and len(tokens[-1].replace(".", "")) <= 2:
        initial = tokens[-1].replace(".", "").strip().lower()[:1]
        last = " ".join(tokens[:-1])
        return f"{last.lower()} {initial}".strip()
    # Sackmann form: first token is the firstname, rest is lastname.
    if len(tokens) >= 2:
        first = tokens[0]
        last = " ".join(tokens[1:])
        return f"{last.lower()} {first[:1].lower()}".strip()
    return s.lower()


def _event_id(tour: str, match_day: date, p1: str, p2: str) -> str:
    """Canonical event_id for one match — uses alphabetically-first as 'home'."""
    home, away, _ = _td_canonical_pair(p1, p2)
    return f"tennis:{tour}:{match_day.isoformat()}:{_norm(home)}-vs-{_norm(away)}".replace(" ", "_")


# ---------------------------------------------------------------------------
# tennis-data.co.uk loader
# ---------------------------------------------------------------------------


def _download_tennis_data_year(tour: str, year: int, timeout: float = 60.0) -> list[dict]:
    import httpx
    import openpyxl  # noqa: F401

    url = ATP_URL_TMPL.format(year=year) if tour == "atp" else WTA_URL_TMPL.format(year=year)
    cache = CACHE_DIR / f"tennis_data_{tour}_{year}.xlsx"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and cache.stat().st_size > 1000:
        data = cache.read_bytes()
    else:
        log.info("downloading tennis-data %s %s ...", tour, year)
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
            data = r.content
            cache.write_bytes(data)
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
    ws = wb.active
    header = None
    out: list[dict] = []
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = list(row)
            continue
        out.append(dict(zip(header, row)))
    return out


def _parse_tennis_data_match(row: dict) -> dict | None:
    """Extract per-match fields we care about.

    Returns dict with: date, winner, loser, home, away, home_won, rank_prob,
    decimal_home, decimal_away (devigged closing prob computed by caller).
    """
    winner = (row.get("Winner") or "").strip()
    loser = (row.get("Loser") or "").strip()
    if not winner or not loser:
        return None
    d = row.get("Date")
    if isinstance(d, datetime):
        match_day = d.date()
    elif isinstance(d, date):
        match_day = d
    else:
        try:
            match_day = datetime.fromisoformat(str(d)).date()
        except Exception:
            return None
    home, away, swap = _td_canonical_pair(winner, loser)
    home_won = home == winner

    # Rank-points BT.
    winner_pts = row.get("WPts")
    loser_pts = row.get("LPts")
    try:
        winner_pts_f = float(winner_pts) if winner_pts is not None else None
        loser_pts_f = float(loser_pts) if loser_pts is not None else None
    except Exception:
        winner_pts_f = loser_pts_f = None
    if swap:
        home_pts, away_pts = loser_pts_f, winner_pts_f
    else:
        home_pts, away_pts = winner_pts_f, loser_pts_f
    rank_prob = _rank_points_prob(home_pts, away_pts)

    # Pinnacle / Bet365 / Avg decimal odds.
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
        except Exception:
            return None

    psw, psl = _f(row.get("PSW")), _f(row.get("PSL"))
    b365w, b365l = _f(row.get("B365W")), _f(row.get("B365L"))
    avgw, avgl = _f(row.get("AvgW")), _f(row.get("AvgL"))

    dec_w = next((x for x in (psw, b365w, avgw) if x is not None and x > 1.0), None)
    dec_l = next((x for x in (psl, b365l, avgl) if x is not None and x > 1.0), None)
    if dec_w is None or dec_l is None:
        decimal_home = decimal_away = None
        market_home = None
    else:
        # winner=home if not swap; loser=home if swap
        if swap:
            decimal_home = dec_l
            decimal_away = dec_w
        else:
            decimal_home = dec_w
            decimal_away = dec_l
        p_h = 1.0 / decimal_home
        p_a = 1.0 / decimal_away
        market_home, _ = _devig_two_way(p_h, p_a)

    return {
        "date": match_day,
        "winner": winner,
        "loser": loser,
        "home": home,
        "away": away,
        "home_won": int(home_won),
        "rank_prob": rank_prob,
        "decimal_home": decimal_home,
        "decimal_away": decimal_away,
        "market_home": market_home,
        "tournament": (row.get("Tournament") or "").strip(),
        "round": (row.get("Round") or "").strip(),
        "surface": (row.get("Surface") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Sackmann Elo loader (walk-forward, per-match)
# ---------------------------------------------------------------------------


def _sackmann_predictions(tour: str, years: list[int], window_start: date, window_end: date):
    """Return list of dicts: {date, home, away, home_prob, home_won, surface, key}.

    The ``_SackmannElo.predictions`` method already yields per-match
    walk-forward predictions in canonical (alphabetically-first as home)
    convention. We just iterate it once across all years.
    """
    engine = _SackmannElo(tour, years=sorted(years), timeout=60.0)
    raw = engine.predictions(window_start, window_end)
    out = []
    for (key, sport, commence, home, away, home_prob, home_won, surf) in raw:
        out.append({
            "key": key,
            "date": commence.date(),
            "home": home,
            "away": away,
            "home_prob": float(home_prob),
            "home_won": int(bool(home_won)),
            "surface": surf,
        })
    return out


# ---------------------------------------------------------------------------
# Cross-source merge helper
# ---------------------------------------------------------------------------


def _merge_key(match_day: date, p1: str, p2: str) -> tuple[date, str, str]:
    """Build a date-and-canonical-name merge key for tennis-data ↔ Sackmann.

    Names get aggressively normalized (lastname + first-initial, lower-cased)
    to absorb the firstname-vs-firstinitial difference.
    """
    a, b, _ = _td_canonical_pair(_norm(p1), _norm(p2))
    return (match_day, a, b)


def _pair_key(p1: str, p2: str) -> tuple[str, str]:
    """Player-pair-only key (no date). Used when Sackmann's ``tourney_date``
    is the tournament start day and tennis-data uses the actual match day,
    so the two sources can't merge on date directly."""
    a, b, _ = _td_canonical_pair(_norm(p1), _norm(p2))
    return (a, b)


# ---------------------------------------------------------------------------
# Backfill driver
# ---------------------------------------------------------------------------


def backfill_tour(tour: str, window_start: date, window_end: date) -> dict[str, int]:
    """Backfill predictions for one tour. Returns per-source row counts."""
    log.info("backfilling %s %s → %s", tour, window_start, window_end)
    years = sorted(set(range(window_start.year, window_end.year + 1)))

    # tennis-data: outcomes + rank-BT + closing odds.
    td_matches: list[dict] = []
    for y in years:
        try:
            rows = _download_tennis_data_year(tour, y)
        except Exception as e:
            log.warning("tennis-data %s %s failed: %s", tour, y, e)
            continue
        for r in rows:
            m = _parse_tennis_data_match(r)
            if m is None:
                continue
            if not (window_start <= m["date"] <= window_end):
                continue
            td_matches.append(m)
    log.info("tennis-data %s: %d matches", tour, len(td_matches))

    # Sackmann Elo: walk-forward per-match preds.
    # Always start the Sackmann load 2 years before window_start so the
    # rating table has enough history before the first scoring match.
    sack_years = sorted(set(range(window_start.year - 2, window_end.year + 1)))
    sack_preds = _sackmann_predictions(tour, sack_years, window_start, window_end)
    log.info("sackmann-%s-elo: %d preds", tour, len(sack_preds))

    # Index Sackmann by (year, player-pair). Sackmann's per-match ``date`` is
    # the tournament-start day, not the actual match day, so we can't merge
    # on date directly. Within a single calendar year a repeat pairing is
    # rare; on the (occasional) repeat, the later upsert wins, which is fine
    # for hold-out coverage purposes.
    sack_index: dict[tuple[int, str, str], dict] = {}
    for s in sack_preds:
        k = (s["date"].year, *_pair_key(s["home"], s["away"]))
        sack_index[k] = s

    pred_rows: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    sack_matched = 0
    # Per-source bet ledger for meta-row computation.
    bet_ledger: list[tuple[date, str, float, float | None, int, int]] = []

    sport: str = tour  # 'atp' or 'wta'

    for m in td_matches:
        eid = _event_id(tour, m["date"], m["home"], m["away"])
        commence = datetime.combine(
            m["date"], datetime.min.time(), tzinfo=timezone.utc,
        ).replace(hour=12).isoformat()
        home_won = int(m["home_won"])
        market_home = m["market_home"]
        home_dec = m["decimal_home"]
        away_dec = m["decimal_away"]
        # See NFL backfill: ``market_close_decimal`` is intentionally None on
        # the predictions table; hold-out ROI is driven by windowed meta rows.
        unified_dec: float | None = None  # noqa: E501 — see NFL backfill comment

        def _record(src: str, hp: float) -> None:
            pred_rows.append({
                "event_id": eid,
                "sport": sport,
                "source": src,
                "commence_time": commence,
                "home": m["home"],
                "away": m["away"],
                "home_prob": float(max(0.001, min(0.999, hp))),
                "home_won": home_won,
                "market_close_home": market_home,
                "market_close_decimal": unified_dec,
            })
            counts[src] += 1
            pick_home = float(hp) >= 0.5
            picked_dec = home_dec if pick_home else away_dec
            picked_won = int((pick_home and home_won == 1) or (not pick_home and home_won == 0))
            bet_ledger.append((m["date"], src, float(hp), picked_dec, picked_won, home_won))

        # 1) tennis-rank-bt
        if m["rank_prob"] is not None:
            _record("tennis-rank-bt", m["rank_prob"])

        # 2) market-close / market-consensus (only when both decimals exist).
        if market_home is not None and home_dec is not None and away_dec is not None:
            _record("market-close", float(market_home))
            _record("market-consensus", float(market_home))

        # 3) sackmann elo — merge by (year, player-pair) since Sackmann's
        # per-match date is tournament-start.
        sack_src_name = f"sackmann-{tour}-elo"
        k = (m["date"].year, *_pair_key(m["home"], m["away"]))
        s = sack_index.get(k)
        if s is not None:
            sack_matched += 1
            # Sackmann's home is alphabetically-first by FULL name; tennis-data
            # canonical home is alphabetically-first by FULL name as well.
            # When normalized keys match, the orientation should match too —
            # but the canonical pair after normalization can flip relative
            # to the full-name canonical, so verify and flip prob if needed.
            sack_home_norm = _norm(s["home"])
            td_home_norm = _norm(m["home"])
            if sack_home_norm == td_home_norm:
                home_prob = s["home_prob"]
            else:
                home_prob = 1.0 - s["home_prob"]
            _record(sack_src_name, float(home_prob))

    log.info("%s: Sackmann matched %d / %d tennis-data matches", tour, sack_matched, len(td_matches))
    log.info("%s: upserting %d prediction rows ...", tour, len(pred_rows))
    upsert_predictions(pred_rows)

    # Per-source meta rows at TWO windows (train, full).
    meta_rows = _build_meta_rows(
        bet_ledger,
        sport=sport,
        train_end=date(2023, 12, 31),
        full_end=window_end,
        window_start=window_start,
    )
    log.info("%s: upserting %d meta rows (one train-window + one full-window per source) ...", tour, len(meta_rows))
    upsert_meta(meta_rows)
    log.info("%s: per-source counts: %s", tour, dict(counts))
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

    See NFL backfill for the full rationale. Emits one row per
    (source, window_end) where window_end ∈ {train_end, full_end}.
    """
    by_src: dict[str, list[tuple[date, float, float | None, int, int]]] = defaultdict(list)
    for d, src, hp, dec, picked_won, home_won in ledger:
        by_src[src].append((d, hp, dec, picked_won, home_won))

    rows = []
    for source, entries in by_src.items():
        for window_end in (train_end, full_end):
            sliced = [e for e in entries if e[0] <= window_end]
            if not sliced:
                continue
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
        default=DEFAULT_START,
    )
    p.add_argument(
        "--end", type=lambda s: datetime.fromisoformat(s).date(),
        default=DEFAULT_END,
    )
    p.add_argument("--tour", choices=("atp", "wta", "both"), default="both")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        import httpx  # noqa: F401
        import openpyxl  # noqa: F401
    except Exception as e:
        log.error("missing dep: %s", e)
        return 2
    tours = ("atp", "wta") if args.tour == "both" else (args.tour,)
    totals: dict[str, int] = {}
    t0 = time.time()
    for tour in tours:
        c = backfill_tour(tour, args.start, args.end)
        for k, v in c.items():
            totals[f"{tour}:{k}"] = v
    log.info("Tennis backfill complete in %.1fs; counts=%s", time.time() - t0, totals)
    log.info("db path: %s", SOURCE_HISTORY_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
