"""Flat $100-per-bet backtest simulator.

This is the headline "is the model profitable" answer Phil asked for. For
every (sport, source) with graded events in ``source_history.db``, we
simulate placing a flat $100 bet on the higher-prob side at the recorded
market_close_decimal, gated by a configurable edge requirement (default
+3pp vs the devigged market close). The blended row uses the post-PR
weights to compute a per-event blended probability and applies the same
edge gate.

Outputs are persisted under ``source_scoreboard.json::backtest_flat_stake``
so the build site can render the headline table at the top of the page.

This module is read-only against ``source_history.db`` — nothing here
mutates ledger state.
"""

from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..config import SOURCE_HISTORY_DB_PATH
from ..model.blend import load_weights, weights_for_sport

log = logging.getLogger(__name__)

# Phil's spec: flat $100 stake on every prediction that clears the edge gate.
FLAT_STAKE: float = 100.0
DEFAULT_EDGE_THRESHOLD: float = 0.03  # +3pp vs devigged market close


@dataclass
class FlatStakeRow:
    sport: str
    source: str
    n_bets: int
    n_wins: int
    stake: float
    profit: float

    @property
    def roi(self) -> float | None:
        if self.stake <= 0:
            return None
        return self.profit / self.stake

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "source": self.source,
            "n_bets": self.n_bets,
            "n_wins": self.n_wins,
            "stake": round(self.stake, 2),
            "profit": round(self.profit, 2),
            "roi": self.roi,
        }


def _decimal_to_devigged_prob(home_dec: float | None, away_dec: float | None) -> tuple[float | None, float | None]:
    """Return (home_devigged_prob, away_devigged_prob) given closing decimals.

    Assumes proportional vig allocation.
    """
    if home_dec is None or away_dec is None:
        return None, None
    if home_dec <= 1.0 or away_dec <= 1.0:
        return None, None
    h = 1.0 / float(home_dec)
    a = 1.0 / float(away_dec)
    total = h + a
    if total <= 0:
        return None, None
    return h / total, a / total


def _simulate_source(
    rows: list[dict],
    *,
    edge_threshold: float,
    stake: float = FLAT_STAKE,
) -> FlatStakeRow:
    """Simulate flat-stake betting for one (sport, source).

    Each row is a prediction with ``home_prob``, ``home_won``, and
    ``market_close_decimal`` on the picked side (the table stores the
    decimal on whichever side the source picked, i.e. higher home_prob).
    The edge gate compares ``home_prob`` to ``1 / market_close_decimal``
    as a single-side implied (no de-vig because we don't have the other
    side here) — that's the standard backfill convention.
    """
    if not rows:
        sport = ""
        source = ""
    else:
        sport = rows[0].get("sport") or ""
        source = rows[0].get("source") or ""
    n_bets = 0
    n_wins = 0
    total_stake = 0.0
    total_profit = 0.0
    for r in rows:
        dec = r.get("market_close_decimal")
        if dec is None or float(dec) <= 1.0:
            continue
        home_prob = float(r["home_prob"])
        pick_home = home_prob >= 0.5
        pick_prob = home_prob if pick_home else 1.0 - home_prob
        implied = 1.0 / float(dec)
        if (pick_prob - implied) < edge_threshold:
            continue
        won = r["home_won"]
        if pick_home:
            won_bet = bool(won)
        else:
            won_bet = not bool(won)
        n_bets += 1
        total_stake += stake
        if won_bet:
            total_profit += stake * (float(dec) - 1.0)
            n_wins += 1
        else:
            total_profit -= stake
    return FlatStakeRow(
        sport=sport,
        source=source,
        n_bets=n_bets,
        n_wins=n_wins,
        stake=total_stake,
        profit=total_profit,
    )


def _simulate_blended(
    rows_by_event: dict[str, list[dict]],
    weights: dict[str, float],
    *,
    edge_threshold: float,
    stake: float = FLAT_STAKE,
    sport: str,
) -> FlatStakeRow:
    """Simulate flat-stake betting using the post-PR weighted blend.

    Two-way market correctness: settles each event at the **picked
    side's** decimal odds (recovered via
    ``flashcat.model.holdout._settlement_decimals``), not at whatever
    decimal the first row happened to carry. See ``_blended_roi`` in
    holdout.py for the rationale; the two simulators share the bug fix.
    """
    from ..model.holdout import _settlement_decimals

    n_bets = 0
    n_wins = 0
    total_stake = 0.0
    total_profit = 0.0
    for event_id, rs in rows_by_event.items():
        home_dec, away_dec, home_won = _settlement_decimals(rs)
        if home_won is None:
            continue
        present = {r["source"]: weights.get(r["source"], 0.0) for r in rs}
        present = {k: v for k, v in present.items() if v > 0}
        if not present:
            continue
        total_w = sum(present.values())
        if total_w <= 0:
            continue
        norm = {k: v / total_w for k, v in present.items()}
        blended = sum(float(r["home_prob"]) * norm.get(r["source"], 0.0) for r in rs)
        blended = max(0.0, min(1.0, blended))
        pick_home = blended >= 0.5
        side_dec = home_dec if pick_home else away_dec
        if side_dec is None or side_dec <= 1.0:
            # No decimal for the picked side; skip rather than settle
            # at the opposite side's price.
            continue
        pick_prob = blended if pick_home else 1.0 - blended
        implied = 1.0 / side_dec
        if (pick_prob - implied) < edge_threshold:
            continue
        won = (pick_home and home_won) or (not pick_home and not home_won)
        n_bets += 1
        total_stake += stake
        if won:
            total_profit += stake * (side_dec - 1.0)
            n_wins += 1
        else:
            total_profit -= stake
    return FlatStakeRow(
        sport=sport, source="flashcat-blended",
        n_bets=n_bets, n_wins=n_wins,
        stake=total_stake, profit=total_profit,
    )


def _meta_based_payload(
    db: Path,
    *,
    stake: float,
    edge_threshold: float,
    weights: dict | None,
) -> dict:
    """Build a flat-stake payload from ``source_history.db.meta``.

    The connectors persist meta rows of the form ``{sport, source, n_bets,
    roi, ...}`` computed against actual closing prices (predictions.
    market_close_decimal isn't always populated by the live build, so we
    treat meta as the source of truth for headline ROI).

    For each (sport, source) we take the LATEST meta row (max window_end),
    set ``stake = n_bets * $100`` and ``profit = roi * stake``. The blended
    row uses the post-PR weights to compute a weighted-average ROI across
    in-blend sources (n_bets = max n_bets among sources; stake/profit
    derived from those).
    """
    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            meta = conn.execute(
                "SELECT sport, source, window_end, n_bets, roi FROM meta"
            ).fetchall()
    except sqlite3.OperationalError:
        return {}

    # Latest row per (sport, source).
    best: dict[tuple[str, str], dict] = {}
    for r in meta:
        d = dict(r)
        key = (d["sport"], d["source"])
        if key not in best or (d.get("window_end") or "") > (best[key].get("window_end") or ""):
            best[key] = d

    per_sport: dict[str, dict] = {}
    total_bets = 0
    total_stake = 0.0
    total_profit = 0.0
    for (sport, source), row in best.items():
        n_bets = int(row.get("n_bets") or 0)
        roi = row.get("roi")
        if n_bets <= 0 or roi is None:
            # Source has no graded bets in any window — still record it as
            # zero so the table can show "n/a" rather than hiding it.
            entry = {
                "sport": sport, "source": source,
                "n_bets": n_bets, "n_wins": 0,
                "stake": 0.0, "profit": 0.0,
                "roi": roi,
            }
            per_sport.setdefault(sport, {"sources": {}, "blended": None})
            per_sport[sport]["sources"][source] = entry
            continue
        st = stake * n_bets
        pf = float(roi) * st
        per_sport.setdefault(sport, {"sources": {}, "blended": None})
        per_sport[sport]["sources"][source] = {
            "sport": sport, "source": source,
            "n_bets": n_bets, "n_wins": 0,  # win count not exposed by meta
            "stake": round(st, 2),
            "profit": round(pf, 2),
            "roi": float(roi),
        }
        total_bets += n_bets
        total_stake += st
        total_profit += pf

    # Blended per-sport: weighted average ROI across in-blend sources.
    for sport, block in per_sport.items():
        sw = weights_for_sport(weights, sport) if weights else {}
        if not sw:
            continue
        pieces: list[tuple[float, float, int]] = []  # (weight, roi, n_bets)
        for src, srow in block["sources"].items():
            if src not in sw:
                continue
            if srow.get("roi") is None or (srow.get("n_bets") or 0) <= 0:
                continue
            pieces.append((sw[src], float(srow["roi"]), int(srow["n_bets"])) )
        if not pieces:
            continue
        w_total = sum(p[0] for p in pieces)
        if w_total <= 0:
            continue
        avg_roi = sum(w * roi for w, roi, _ in pieces) / w_total
        # Bet count: events where at least one in-blend source had a graded
        # bet — conservatively the max across sources.
        b_n_bets = max(n for _, _, n in pieces)
        b_stake = stake * b_n_bets
        b_profit = avg_roi * b_stake
        block["blended"] = {
            "sport": sport, "source": "flashcat-blended",
            "n_bets": b_n_bets, "n_wins": 0,
            "stake": round(b_stake, 2),
            "profit": round(b_profit, 2),
            "roi": avg_roi,
            "roi_source": "weighted_per_source_meta",
        }

    totals = {
        "n_bets": total_bets,
        "stake": round(total_stake, 2),
        "profit": round(total_profit, 2),
        "roi": (total_profit / total_stake) if total_stake > 0 else None,
    }
    return {
        "edge_threshold": edge_threshold,
        "stake": stake,
        "per_sport": per_sport,
        "totals": totals,
        "source": "meta",
    }


def run_flat_stake_backtest(
    db_path: Path | None = None,
    *,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    stake: float = FLAT_STAKE,
    weights: dict | None = None,
) -> dict:
    """Run the flat $100 simulator for every sport/source with graded data.

    Returns a payload of shape::

        {
          "edge_threshold": 0.03,
          "stake": 100.0,
          "per_sport": {
            sport: {
              "blended": {n_bets, n_wins, stake, profit, roi},
              "sources": {source: {n_bets, n_wins, stake, profit, roi}, ...},
            }, ...
          },
          "totals": {n_bets, stake, profit, roi},
        }

    Prefers per-prediction simulation when ``predictions.market_close_decimal``
    is populated; falls back to ``meta.n_bets * roi`` reconstruction
    otherwise.
    """
    db = db_path or SOURCE_HISTORY_DB_PATH
    if not Path(db).exists():
        return {}

    if weights is None:
        weights = load_weights()

    try:
        with sqlite3.connect(str(db)) as conn:
            conn.row_factory = sqlite3.Row
            preds = conn.execute(
                """
                SELECT event_id, sport, source, home_prob, home_won,
                       market_close_home, market_close_decimal
                FROM predictions
                WHERE home_won IS NOT NULL
                  AND market_close_decimal IS NOT NULL
                """,
            ).fetchall()
    except sqlite3.OperationalError:
        return {}

    if not preds:
        # No predictions with settlement prices — fall back to meta.
        return _meta_based_payload(
            db, stake=stake, edge_threshold=edge_threshold, weights=weights,
        )

    # Group by sport → source for per-source sims, sport → event for blend.
    by_sport_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_sport_event: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in preds:
        d = dict(r)
        by_sport_source[(d["sport"], d["source"])].append(d)
        by_sport_event[d["sport"]][d["event_id"]].append(d)

    per_sport: dict[str, dict] = {}
    total_bets = 0
    total_stake = 0.0
    total_profit = 0.0
    for (sport, source), rows in by_sport_source.items():
        sim = _simulate_source(rows, edge_threshold=edge_threshold, stake=stake)
        per_sport.setdefault(sport, {"sources": {}, "blended": None})
        per_sport[sport]["sources"][source] = sim.to_dict()
        total_bets += sim.n_bets
        total_stake += sim.stake
        total_profit += sim.profit

    for sport, events in by_sport_event.items():
        sw = weights_for_sport(weights, sport) if weights else {}
        if sw:
            blended = _simulate_blended(
                events, sw,
                edge_threshold=edge_threshold,
                stake=stake, sport=sport,
            )
            per_sport.setdefault(sport, {"sources": {}, "blended": None})
            per_sport[sport]["blended"] = blended.to_dict()

    totals = {
        "n_bets": total_bets,
        "stake": round(total_stake, 2),
        "profit": round(total_profit, 2),
        "roi": (total_profit / total_stake) if total_stake > 0 else None,
    }
    # If per-prediction sim turned up no bets (e.g. only one sport had
    # market_close_decimal populated), backfill the other sports from meta.
    if total_bets == 0:
        return _meta_based_payload(
            db, stake=stake, edge_threshold=edge_threshold, weights=weights,
        )
    meta_fallback = _meta_based_payload(
        db, stake=stake, edge_threshold=edge_threshold, weights=weights,
    )
    if meta_fallback:
        # Fill in sports that the per-prediction sim missed.
        for sport, block in (meta_fallback.get("per_sport") or {}).items():
            if sport not in per_sport:
                per_sport[sport] = block
                bm = block.get("blended") or {}
                total_bets += int(bm.get("n_bets") or 0)
                total_stake += float(bm.get("stake") or 0)
                total_profit += float(bm.get("profit") or 0)
        totals = {
            "n_bets": total_bets,
            "stake": round(total_stake, 2),
            "profit": round(total_profit, 2),
            "roi": (total_profit / total_stake) if total_stake > 0 else None,
        }
    return {
        "edge_threshold": edge_threshold,
        "stake": stake,
        "per_sport": per_sport,
        "totals": totals,
        "source": "predictions",
    }


def format_flat_stake_table(payload: dict) -> str:
    """Render the headline table (Phil's preferred view)."""
    if not payload:
        return "(no flat-stake data \u2014 source_history.db has no settled-with-prices rows)"
    per_sport = payload.get("per_sport") or {}
    if not per_sport:
        totals = payload.get("totals") or {}
        return (
            "Sport   n_bets   Stake     Profit     ROI\n"
            "----------------------------------------\n"
            f"TOTALS  {totals.get('n_bets', 0):>7}  "
            f"${totals.get('stake', 0):>10,.0f}  "
            f"${totals.get('profit', 0):>10,.0f}  "
            f"{(totals.get('roi') or 0)*100:>+6.2f}%"
        )
    header = (
        f"{'Sport':<6} {'Source':<32} {'n_bets':>7}  "
        f"{'Stake':>12}  {'Profit':>12}  {'ROI':>9}"
    )
    lines = [header, "-" * len(header)]
    for sport in sorted(per_sport.keys()):
        block = per_sport[sport]
        bm = block.get("blended")
        sport_label = sport.upper()
        # Always emit a leading sport-labeled row so the table doesn't
        # "borrow" a sport label from the previous block when there's no
        # blended row (e.g. sports where no in-blend source has graded ROI).
        if bm:
            roi = bm.get("roi")
            roi_s = f"{roi*100:+.2f}%" if roi is not None else "n/a"
            lines.append(
                f"{sport_label:<6} {'flashcat-blended':<32} "
                f"{bm.get('n_bets', 0):>7}  "
                f"${bm.get('stake', 0):>10,.0f}  "
                f"${bm.get('profit', 0):>10,.0f}  {roi_s:>9}"
            )
            sport_label = ""
        for src, srow in sorted(
            (block.get("sources") or {}).items(),
            key=lambda kv: -(kv[1].get("n_bets") or 0),
        ):
            roi = srow.get("roi")
            roi_s = f"{roi*100:+.2f}%" if roi is not None else "n/a"
            lines.append(
                f"{sport_label:<6} {src:<32} "
                f"{srow.get('n_bets', 0):>7}  "
                f"${srow.get('stake', 0):>10,.0f}  "
                f"${srow.get('profit', 0):>10,.0f}  {roi_s:>9}"
            )
            sport_label = ""
    totals = payload.get("totals") or {}
    roi = totals.get("roi")
    roi_s = f"{roi*100:+.2f}%" if roi is not None else "n/a"
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<6} {'(all sports, all sources)':<32} "
        f"{totals.get('n_bets', 0):>7}  "
        f"${totals.get('stake', 0):>10,.0f}  "
        f"${totals.get('profit', 0):>10,.0f}  {roi_s:>9}"
    )
    return "\n".join(lines)
