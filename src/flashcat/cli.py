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

from .backtest.flat_stake import format_flat_stake_table, run_flat_stake_backtest
from .backtest.runner import run_backtest, run_multi_sport_backtest
from .backtest.scoreboard_patch import patch_scoreboard
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
    CFBCfbfastREPA,
    CFBESPNFPI,
    CFBMarketConsensus,
    ESPNScoreboard,
    FanDuel,
    MLBStatcastLineup,
    MLBWeather,
    PGADatagolf,
    PGAESPNScoreboard,
    PGAMarketConsensus,
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
        if isinstance(c, (PGADatagolf, PGAESPNScoreboard, PGAMarketConsensus)):
            # PGA connectors gate themselves on data availability
            # (DATAGOLF_API_KEY, active ESPN scoreboard, Odds API outright
            # markets). We always advertise ``pga`` here so the build loop
            # fetches from them; per-sport mode gating (RESEARCH MODE)
            # ensures negative-ROI PGA picks never display $ stakes.
            discovered.add("pga")
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
        # PGA connectors (PR #15). DataGolf SG model (key-gated, free-tier
        # endpoints only) + ESPN PGA leaderboard proxy + Odds API outright
        # winner consensus. All three return [] off-season / unkeyed so
        # they're cheap to wire in year-round.
        PGADatagolf(),
        PGAESPNScoreboard(),
        PGAMarketConsensus(),
        # CFB connectors (PR #14). EPA + market consensus + FPI predictor.
        # All return [] outside CFB season so they're cheap to wire in year-round.
        CFBCfbfastREPA(),
        CFBMarketConsensus(),
        CFBESPNFPI(),
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
    """Fit per-sport Platt scaling against the post-exclusion blend.

    Re-blends ``source_history.db.predictions`` using the *current*
    ``data/source_weights.json`` (i.e. after the de-dilution PR's exclusion
    + sharper-β reweighter has run) and fits
    σ(α + β · logit(p)) on the (post-exclusion blended_prob, outcome)
    pairs. Persists per-sport coefficients to ``data/calibration.json``.

    Falls back to the legacy scoreboard-rows path when source_history.db is
    empty or missing.
    """
    import json
    from .config import SOURCE_HISTORY_DB_PATH, SOURCE_SCOREBOARD_PATH
    from .model.blend import load_weights, weights_for_sport
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    per_sport: dict[str, dict] = {}

    # Preferred path: re-blend predictions out of source_history.db.
    used_db = False
    if SOURCE_HISTORY_DB_PATH.exists():
        try:
            import sqlite3
            weights = load_weights()
            with sqlite3.connect(str(SOURCE_HISTORY_DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                sports = [
                    r[0]
                    for r in conn.execute(
                        "SELECT DISTINCT sport FROM predictions WHERE home_won IS NOT NULL"
                    ).fetchall()
                ]
                for sport in sports:
                    sw = weights_for_sport(weights, sport)
                    if not sw:
                        continue
                    rows = conn.execute(
                        "SELECT event_id, source, home_prob, home_won FROM predictions "
                        "WHERE sport=? AND home_won IS NOT NULL", (sport,),
                    ).fetchall()
                    by_event: dict[str, list[dict]] = {}
                    for r in rows:
                        if r["source"] not in sw:
                            continue
                        by_event.setdefault(r["event_id"], []).append(dict(r))
                    cal_pairs: list[tuple[float, bool]] = []
                    for ev_id, rs in by_event.items():
                        present = {r["source"]: sw[r["source"]] for r in rs}
                        total = sum(present.values())
                        if total <= 0:
                            continue
                        norm = {k: v / total for k, v in present.items()}
                        blended = sum(
                            float(r["home_prob"]) * norm[r["source"]] for r in rs
                        )
                        blended = max(0.0, min(1.0, blended))
                        outcome = bool(rs[0]["home_won"])
                        cal_pairs.append((blended, outcome))
                    if not cal_pairs:
                        continue
                    used_db = True
                    fit = fit_platt(cal_pairs)
                    if not fit:
                        log.info(
                            "%s: calibration not fit (n=%d) — will pass-through",
                            sport, len(cal_pairs),
                        )
                        continue
                    alpha, beta = fit
                    per_sport[sport] = {
                        "alpha": alpha, "beta": beta, "n": len(cal_pairs),
                    }
                    log.info(
                        "%s: Platt fit (post-exclusion) α=%.3f β=%.3f (n=%d)",
                        sport, alpha, beta, len(cal_pairs),
                    )
        except Exception as exc:
            log.warning("calibrate: source_history.db path failed (%s) — falling back", exc)

    # Fallback: legacy scoreboard path (pre-PR-19 behaviour).
    if not used_db:
        if not SOURCE_SCOREBOARD_PATH.exists():
            log.info("No scoreboard yet — skipping calibration.")
            return
        with open(SOURCE_SCOREBOARD_PATH) as f:
            sb = json.load(f)
        for sport, p in (sb.get("per_sport") or {}).items():
            if sport in per_sport:
                continue
            bm = (p or {}).get("blended") or {}
            cal = bm.get("calibration_rows") or []
            if not cal:
                continue
            fit = fit_platt([(float(prob), bool(y)) for prob, y in cal])
            if not fit:
                log.info("%s: calibration not fit (n=%d) — will pass-through", sport, len(cal))
                continue
            alpha, beta = fit
            per_sport[sport] = {"alpha": alpha, "beta": beta, "n": len(cal)}
            log.info("%s: Platt fit α=%.3f β=%.3f (n=%d)", sport, alpha, beta, len(cal))
    save_coefficients(per_sport)


@app.command("holdout")
def holdout(
    output: str = typer.Option(
        "", help="Optional path to write the table as text (default: stdout only)"
    ),
) -> None:
    """Run walk-forward hold-out validation and print the per-sport table.

    Splits ``data/source_history.db`` into 2022-2023 (training) and 2024
    (held-out). Fits the de-dilution reweighter on training-window source
    stats, applies the frozen weights to the held-out predictions, and
    reports training vs hold-out ROI.
    """
    from .model.holdout import format_holdout_table, run_holdout_validation
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    results = run_holdout_validation()
    table = format_holdout_table(results)
    typer.echo(table)
    if output:
        Path(output).write_text(table + "\n")
        log.info("Wrote holdout table to %s", output)


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
    excluded = (payload.get("excluded") or {})
    for sport, ex_list in excluded.items():
        if not ex_list:
            continue
        log.info("Excluded from %s pool:", sport.upper())
        for ex in ex_list:
            src = ex.get("source") or "(min-sources fallback)"
            log.info("  %-40s  %s", src, ex.get("reason", ""))


@app.command("flat-stake")
def flat_stake() -> None:
    """Run the headline flat-$100 backtest and print the per-sport table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    payload = run_flat_stake_backtest()
    typer.echo(format_flat_stake_table(payload))


@app.command("patch-scoreboard")
def patch_scoreboard_cmd() -> None:
    """Patch source_scoreboard.json with addendum-10 fixes + flat-stake table.

    Run AFTER reweight (so the in-blend weights reflect the post-exclusion
    pool) and BEFORE build (so the rendered site sees the patched data).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    flat = run_flat_stake_backtest()
    sb = patch_scoreboard(flat_stake_payload=flat)
    if not sb:
        log.info("No scoreboard to patch.")
        return
    log.info("Patched source_scoreboard.json.")


@app.command("source-accountability")
def source_accountability_cmd(
    out_dir: str = typer.Option(
        "paw-reports/sportsbetting",
        help="Where to write the report (latest + dated archive).",
    ),
    skip_tennis_per_event: bool = typer.Option(
        False,
        help="Skip the tennis-data.co.uk re-pull (offline mode).",
    ),
    tennis_start: int = typer.Option(2022, help="First year for tennis per-event ledger."),
    tennis_end: int = typer.Option(2024, help="Last year (inclusive) for tennis per-event ledger."),
) -> None:
    """Per-source accountability report — Phil's no-spin source audit.

    Walks source_history.db + tennis-data.co.uk archives + predict.tennis
    self-report, scores every (sport, source) on Brier / log loss / hit rate
    / $100-flat ROI / CLV / drawdown / longest losing streak, and writes a
    markdown + JSON report under ``paw-reports/sportsbetting/``.

    This is the FIRST RUN of a recurring weekly process. See
    ``docs/AGENT_LOOP.md`` for the standing cadence.
    """
    from pathlib import Path
    from .source_accountability import assemble_report, write_report

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    report = assemble_report(
        tennis_years=range(int(tennis_start), int(tennis_end) + 1),
        include_tennis_per_event=not skip_tennis_per_event,
    )
    dated, latest = write_report(report, out_dir=Path(out_dir))
    typer.echo(f"wrote {dated}")
    typer.echo(f"wrote {latest}")
    typer.echo(f"n_sources={report['n_sources']}")


@app.command()
def all(
    start: str = typer.Option(""),
    end: str = typer.Option(""),
    sport: str = typer.Option("all"),
    days_ahead: int = typer.Option(2),
) -> None:
    """Backtest → reweight → patch-scoreboard → calibrate → build."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    backtest(start=start, end=end, sport=sport)
    reweight()
    patch_scoreboard_cmd()
    calibrate()
    build(days_ahead=days_ahead)


if __name__ == "__main__":
    app()
