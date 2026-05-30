"""Post-processor to fix aggregation gaps in ``source_scoreboard.json``.

Phil's bug report (addendum item 10):

* ``per_sport[sport].sources[source].n_bets`` is ``None`` for every source
  the backtest didn't actually place a bet on. We can fill it in from
  ``data/source_history.db.meta.n_bets``.
* ``per_sport[sport].blended.roi`` is ``None`` for MLB / NBA / CFB / PGA
  even when underlying per-source meta rows DO have graded ROI. We can
  synthesize ``blended.roi`` as a weight-weighted average of per-source
  meta ROIs using the post-PR softmax weights, flagging it with
  ``roi_source = "weighted_per_source_meta"`` so downstream consumers
  know it's derived.
* The new ``backtest_flat_stake`` section (addendum item 11) needs to be
  stitched in here too so the build site can render the headline table.

This module mutates the loaded scoreboard dict in-place and rewrites the
JSON file. It is idempotent — running it twice yields the same payload.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..config import SOURCE_HISTORY_DB_PATH, SOURCE_SCOREBOARD_PATH
from ..model.blend import load_weights, weights_for_sport

log = logging.getLogger(__name__)


def _latest_meta_per_source(db_path: Path) -> dict[tuple[str, str], dict]:
    """Return ``{(sport, source): meta_row}`` with the row having the most n_events."""
    if not Path(db_path).exists():
        return {}
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT sport, source, n_events, n_bets, roi, brier FROM meta"
            ).fetchall()
    except sqlite3.OperationalError:
        return {}
    best: dict[tuple[str, str], dict] = {}
    for r in rows:
        d = dict(r)
        key = (d["sport"], d["source"])
        cur = best.get(key)
        if cur is None or (d.get("n_events") or 0) > (cur.get("n_events") or 0):
            best[key] = d
    return best


def patch_scoreboard(
    scoreboard_path: Path | None = None,
    db_path: Path | None = None,
    *,
    weights: dict | None = None,
    flat_stake_payload: dict | None = None,
) -> dict:
    """Patch the on-disk scoreboard. Returns the patched dict.

    No-op when the file doesn't exist. Persists the patched JSON back to
    disk.
    """
    sp = scoreboard_path or SOURCE_SCOREBOARD_PATH
    if not Path(sp).exists():
        return {}
    with open(sp) as f:
        sb = json.load(f)
    if not isinstance(sb, dict):
        return {}

    if weights is None:
        weights = load_weights()
    meta_by_key = _latest_meta_per_source(db_path or SOURCE_HISTORY_DB_PATH)

    per_sport = sb.get("per_sport") or {}
    # Pre-collect every (sport, source) that exists in meta so we can
    # inject sources the in-memory backtest missed entirely.
    sports_in_meta = sorted({s for s, _ in meta_by_key})
    for sport in sports_in_meta:
        if sport not in per_sport:
            per_sport[sport] = {"n_events": 0, "sources": {}, "blended": None}
    sb["per_sport"] = per_sport

    for sport, block in per_sport.items():
        if not isinstance(block, dict):
            continue
        sources = block.get("sources") or {}
        block["sources"] = sources
        # (a0) Inject sources present in meta but missing from the scoreboard
        # (e.g. mlb-statcast-lineup, nba-bref-srs-pace populated via the
        # backfill scripts but not the in-memory backtest).
        for (m_sport, m_source), meta in meta_by_key.items():
            if m_sport != sport or m_source in sources:
                continue
            sources[m_source] = {
                "n_events": int(meta.get("n_events") or 0),
                "n_bets": int(meta.get("n_bets") or 0) or None,
                "brier": meta.get("brier"),
                "roi": meta.get("roi"),
                "wins": 0,
                "losses": 0,
                "wagered": 0.0,
                "profit": 0.0,
                "roi_source": "source_history.meta",
            }
        # (a) Populate n_bets from meta for every existing source row.
        for source, srow in sources.items():
            if not isinstance(srow, dict):
                continue
            meta = meta_by_key.get((sport, source))
            if meta is None:
                continue
            # Only fill missing fields — never overwrite values the
            # in-memory backtest actually produced.
            if srow.get("n_bets") is None or srow.get("n_bets") == 0:
                srow["n_bets"] = int(meta.get("n_bets") or 0) or None
            if srow.get("roi") is None and meta.get("roi") is not None:
                srow["roi"] = float(meta["roi"])
                srow["roi_source"] = "source_history.meta"
            if srow.get("n_events") in (None, 0):
                srow["n_events"] = int(meta.get("n_events") or 0)

        # (b) Synthesize blended.roi when missing but underlying meta has it.
        blended = block.get("blended")
        if blended is None or blended.get("roi") is None:
            sw = weights_for_sport(weights, sport)
            pieces: list[tuple[float, float, int]] = []
            for src, w in sw.items():
                srow = sources.get(src)
                if not srow:
                    continue
                roi = srow.get("roi")
                n_bets = srow.get("n_bets") or 0
                if roi is None or n_bets < 50:
                    continue
                pieces.append((float(w), float(roi), int(n_bets)))
            if pieces:
                w_total = sum(p[0] for p in pieces)
                if w_total > 0:
                    synth_roi = sum(w * r for w, r, _ in pieces) / w_total
                    synth_n = max(n for _, _, n in pieces)
                    block["blended"] = dict(blended or {})
                    block["blended"]["roi"] = synth_roi
                    block["blended"]["n_bets"] = synth_n
                    block["blended"]["n_events"] = synth_n
                    block["blended"]["roi_source"] = "weighted_per_source_meta"
                    log.info(
                        "scoreboard_patch: %s blended.roi = %.4f (weighted "
                        "from %d source(s), n=%d)",
                        sport, synth_roi, len(pieces), synth_n,
                    )

    # (c) Stitch in the flat-stake payload (addendum item 11).
    if flat_stake_payload is not None:
        sb["backtest_flat_stake"] = flat_stake_payload

    with open(sp, "w") as f:
        json.dump(sb, f, indent=2)
    return sb
