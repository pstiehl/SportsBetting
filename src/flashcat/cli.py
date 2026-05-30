"""Flashcat CLI — `python -m flashcat`.

Commands:
  - build:       pull today's slate, blend, write index/site
  - backtest:    run historical backtest, write source_scoreboard.json
  - reweight:    softmax over -Brier, update data/source_weights.json
  - all:         backtest → reweight → build
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import typer

from .backtest.runner import run_backtest, run_multi_sport_backtest
from .build_site import build as build_site
from .config import (
    NoLiveDataError,
    backtest_end,
    backtest_start,
    ensure_dirs,
    use_samples_fallback,
)
from .db import init_db
from .model.blend import blend_events, load_weights
from .model.calibration import fit_platt, save_coefficients
from .model.reweight import update_weights as update_weights_fn
from .signals.favlong import detect as detect_favlong
from .signals.sharp import detect as detect_sharp
from .sources import (
    Bovada,
    ESPNScoreboard,
    FanDuel,
    MLBStatcastLineup,
    MLBWeather,
    Polymarket,
    TheOddsAPI,
)
from .types import SPORTS, Event, Sport

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Flashcat Betting CLI")


def _merge_events(*lists: list[Event]) -> list[Event]:
    """Merge events from multiple connectors by (sport, home, away, date).

    The first list to provide a particular team-key wins event_id, but probs
    and lines from subsequent connectors are appended.
    """
    by_key: dict[tuple, Event] = {}
    for lst in lists:
        for ev in lst:
            key = (
                ev.sport,
                _normalize(ev.home),
                _normalize(ev.away),
                ev.commence_time.date().isoformat(),
            )
            # Also try the swapped key (player ordering varies across sources).
            swapped = (
                ev.sport,
                _normalize(ev.away),
                _normalize(ev.home),
                ev.commence_time.date().isoformat(),
            )
            if key in by_key:
                target = by_key[key]
                target.source_probs.extend(ev.source_probs)
                target.lines.extend(ev.lines)
            elif swapped in by_key:
                # Probability semantics: source_probs are home_win_prob.
                # Flipping requires us to invert each source prob (1 - p).
                target = by_key[swapped]
                for sp in ev.source_probs:
                    target.source_probs.append(
                        type(sp)(
                            source=sp.source,
                            home_win_prob=max(
                                0.001, min(0.999, 1.0 - sp.home_win_prob)
                            ),
                            captured_at=sp.captured_at,
                            notes=f"{sp.notes} (inverted to match home/away)".strip(),
                        )
                    )
                # Lines also need to flip sides.
                for ln in ev.lines:
                    target.lines.append(
                        type(ln)(
                            book=ln.book,
                            side=("home" if ln.side == "away" else "away"),
                            american=ln.american,
                            captured_at=ln.captured_at,
                            is_opening=ln.is_opening,
                        )
                    )
            else:
                by_key[key] = ev
    return list(by_key.values())


def _normalize(name: str) -> str:
    return name.lower().replace(".", "").replace("the ", "").strip()


def _active_sports(connectors) -> list[Sport]:
    """Discover in-season sports today.

    Order:
      1. Ask Odds API ``active_sports()`` if a key is configured and reachable.
      2. Always union in atp/wta when Bovada is in the connector list — Bovada
         discovers active tennis tournaments live, so as long as Roland Garros
         (or any other tour) is running we want those events in the slate.
      3. Fall back to the full SPORTS list otherwise. Per-source fetch yields
         0 events for out-of-season sports; the fail-loud check trips if
         nothing comes back.
    """
    discovered: set[Sport] = set()
    for c in connectors:
        if isinstance(c, TheOddsAPI) and c.api_key:
            try:
                sports = c.active_sports()
            except Exception:  # noqa: BLE001
                sports = []
            for tag, _key in sports:
                discovered.add(tag)
        if isinstance(c, Bovada):
            # Bovada always handles tennis discovery itself.
            discovered.update({"atp", "wta"})
    if discovered:
        return sorted(discovered)
    return list(SPORTS)


@app.command()
def build(
    days_ahead: int = typer.Option(2, help="How many days of upcoming events to include"),
) -> None:
    """Pull today + N days of events from live connectors, blend, build site.

    Fails loud (NoLiveDataError) if every live source returned 0 events for
    every in-season sport. Set FLASHCAT_USE_SAMPLES=1 to bypass for local dev.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ensure_dirs()
    init_db()
    start = date.today()
    end = start + timedelta(days=days_ahead)
    log.info("Building slate for %s → %s", start, end)
    connectors = [
        TheOddsAPI(),
        Bovada(),
        FanDuel(),
        ESPNScoreboard(),
        Polymarket(),
        MLBStatcastLineup(),
        MLBWeather(),
    ]

    active = _active_sports(connectors)
    log.info("Active sports today: %s", active)

    all_lists = []
    per_sport_counts: dict[str, int] = {s: 0 for s in active}
    for c in connectors:
        try:
            lst = c.fetch_events(start, end)
            log.info("  %s → %d events", c.name, len(lst))
            for ev in lst:
                if ev.sport in per_sport_counts:
                    per_sport_counts[ev.sport] += 1
            all_lists.append(lst)
        except Exception as e:  # noqa: BLE001
            log.warning("  %s failed: %s", c.name, e)
    events = _merge_events(*all_lists)
    # Restrict to in-season sports only — don't render Sept NFL games on May 29.
    events = [e for e in events if e.sport in active]

    if not events:
        if use_samples_fallback():
            log.warning(
                "No live events but FLASHCAT_USE_SAMPLES=1 → rendering empty slate"
            )
        else:
            raise NoLiveDataError(
                f"No live events for any in-season sport ({active}). "
                "Refusing to ship stale samples. Set FLASHCAT_USE_SAMPLES=1 "
                "for offline local builds, or wait for sources to recover."
            )

    # Per-sport coverage check: if a sport is "active" but no source returned
    # anything for it, that's worth logging loudly (not fatal — sport might be
    # in season but in an off-day).
    for s, n in per_sport_counts.items():
        log.info("  in-season %s coverage: %d raw events from live sources", s, n)

    # If an event has no source probs but does have lines, synthesize a market-close source prob.
    from .backtest.runner import _attach_market_source_prob
    _attach_market_source_prob(events)
    weights = load_weights()
    blended = blend_events(events, weights)
    for ev in blended:
        chalk = detect_favlong(ev)
        if chalk:
            ev.signals.append(chalk)
        ev.signals.extend(detect_sharp(ev))
    log.info("Building site with %d events", len(blended))
    build_site(blended)


@app.command()
def backtest(
    start: str = typer.Option(
        "", help="Start date YYYY-MM-DD (default FLASHCAT_BACKTEST_START or 2022-01-01)"
    ),
    end: str = typer.Option(
        "", help="End date YYYY-MM-DD (default FLASHCAT_BACKTEST_END or today)"
    ),
    sport: str = typer.Option(
        "all", help="Sport (nfl, nba, mlb, atp, wta, or 'all' for multi-sport)"
    ),
) -> None:
    """Run historical backtest and write source_scoreboard.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ensure_dirs()
    init_db()
    s = date.fromisoformat(start or backtest_start())
    e = date.fromisoformat(end or backtest_end())
    if sport == "all":
        run_multi_sport_backtest(s, e)
    else:
        run_backtest(s, e, sport=sport)


@app.command()
def calibrate() -> None:
    """Fit per-sport Platt scaling on the latest backtest predictions.

    Reads ``data/source_scoreboard.json`` (per_sport → blended.calibration_data
    style) and the blended events' (prob, outcome) pairs to fit
    σ(α + β · logit(p)). Persists to ``data/calibration.json``.
    """
    import json
    from .config import SOURCE_SCOREBOARD_PATH
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    if not SOURCE_SCOREBOARD_PATH.exists():
        log.info("No scoreboard yet — skipping calibration.")
        return
    with open(SOURCE_SCOREBOARD_PATH) as f:
        sb = json.load(f)
    per_sport: dict[str, dict] = {}
    for sport, p in (sb.get("per_sport") or {}).items():
        bm = (p or {}).get("blended") or {}
        cal = bm.get("calibration_rows") or []
        if not cal:
            # Fall back to calibration_bins midpoints × actual hit rate
            continue
        fit = fit_platt([(float(prob), bool(y)) for prob, y in cal])
        if not fit:
            log.info("%s: calibration not fit (n=%d) — will pass-through", sport, len(cal))
            continue
        alpha, beta = fit
        per_sport[sport] = {"alpha": alpha, "beta": beta, "n": len(cal)}
        log.info("%s: Platt fit α=%.3f β=%.3f (n=%d)", sport, alpha, beta, len(cal))
    save_coefficients(per_sport)


@app.command()
def reweight() -> None:
    """Update source weights from the latest scoreboard."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ensure_dirs()
    init_db()
    payload = update_weights_fn()
    if not payload:
        log.info("No eligible sources for reweighting (need ≥ 50 events).")
        return
    log.info("Weight mode: %s", payload.get("mode", "?"))
    log.info("Global pool:")
    for k, v in sorted((payload.get("global") or {}).items(), key=lambda kv: -kv[1]):
        log.info("  %-40s  %6.1f%%", k, v * 100)
    for sport, pool in (payload.get("by_sport") or {}).items():
        log.info("Per-sport pool (%s):", sport.upper())
        for k, v in sorted(pool.items(), key=lambda kv: -kv[1]):
            log.info("  %-40s  %6.1f%%", k, v * 100)


@app.command()
def all(
    start: str = typer.Option(""),
    end: str = typer.Option(""),
    sport: str = typer.Option("all"),
    days_ahead: int = typer.Option(2),
) -> None:
    """Backtest → reweight → calibrate → build today's slate → render site."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    backtest(start=start, end=end, sport=sport)
    reweight()
    calibrate()
    build(days_ahead=days_ahead)


if __name__ == "__main__":
    app()
