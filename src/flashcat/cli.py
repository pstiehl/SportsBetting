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

from .backtest.runner import run_backtest
from .build_site import build as build_site
from .config import ensure_dirs
from .db import init_db
from .model.blend import blend_events, load_weights
from .model.reweight import update_weights as update_weights_fn
from .signals.favlong import detect as detect_favlong
from .signals.sharp import detect as detect_sharp
from .sources import (
    ESPNScoreboard,
    Polymarket,
    TheOddsAPI,
)
from .types import Event

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
            if key in by_key:
                existing = by_key[key]
                existing.source_probs.extend(ev.source_probs)
                existing.lines.extend(ev.lines)
            else:
                by_key[key] = ev
    return list(by_key.values())


def _normalize(name: str) -> str:
    return name.lower().replace(".", "").replace("the ", "").strip()


@app.command()
def build(
    days_ahead: int = typer.Option(2, help="How many days of upcoming events to include"),
) -> None:
    """Pull today + N days of events from live connectors, blend, build site."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ensure_dirs()
    init_db()
    start = date.today()
    end = start + timedelta(days=days_ahead)
    log.info("Building slate for %s → %s", start, end)
    connectors = [TheOddsAPI(), ESPNScoreboard(), Polymarket()]
    all_lists = []
    for c in connectors:
        try:
            lst = c.fetch_events(start, end)
            log.info("  %s → %d events", c.name, len(lst))
            all_lists.append(lst)
        except Exception as e:  # noqa: BLE001
            log.warning("  %s failed: %s", c.name, e)
    events = _merge_events(*all_lists)
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
    start: str = typer.Option("2023-09-01", help="Start date YYYY-MM-DD"),
    end: str = typer.Option("2024-02-15", help="End date YYYY-MM-DD"),
    sport: str = typer.Option("nfl", help="Sport"),
) -> None:
    """Run historical backtest and write source_scoreboard.json."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ensure_dirs()
    init_db()
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    run_backtest(s, e, sport=sport)


@app.command()
def reweight() -> None:
    """Update source weights from the latest scoreboard."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    new = update_weights_fn()
    if not new:
        log.info("No eligible sources for reweighting (need ≥ 20 events).")
    else:
        for k, v in sorted(new.items(), key=lambda kv: -kv[1]):
            log.info("  %-22s  %6.1f%%", k, v * 100)


@app.command()
def all(
    start: str = typer.Option("2023-09-01"),
    end: str = typer.Option("2024-02-15"),
    sport: str = typer.Option("nfl"),
    days_ahead: int = typer.Option(2),
) -> None:
    """Backtest → reweight → build today's slate → render site."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    backtest(start=start, end=end, sport=sport)
    reweight()
    build(days_ahead=days_ahead)


if __name__ == "__main__":
    app()
