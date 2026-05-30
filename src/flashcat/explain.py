"""Per-pick "Why this pick" rationale generator.

For each event we surface the top-3 signals driving the model's pick. The
rendered event card embeds these inside a ``<details>`` block on the live
site so Phil can audit the decision without parsing source-by-source
probabilities.

Priority order (highest leverage first):
  1. Statcast per-batter matchup mismatches (MLB only) — surfaces specific
     batters when their xwOBA deviates meaningfully from league mean, otherwise
     falls back to a generic lineup-edge string.
  2. Weather / park run-environment delta (MLB only).
  3. Per-sport flagship signal (NFL: EPA differential, NBA: SRS diff).
  4. Market consensus vs blended probability — the "edge" statement.
  5. Active signal badges (chalk-overpriced, RLM, book-dispersion).

The output is a list of plain-text strings, ordered by priority, designed
to render directly inside the event card. The caller decides truncation.
"""

from __future__ import annotations

import re
from typing import Iterable

from .types import Event, SourceProb, american_to_prob, devig_two_way

# Per-batter inclusion threshold mirrors the source connector — batters
# whose xwOBA deviates from league mean by less than this absolute value
# fall back to the generic "Statcast lineup edge" rationale string.
BATTER_RATIONALE_DEVIATION_THRESHOLD = 0.030
# Rough conversion from a single batter's PA-weighted xwOBA delta to runs
# per game. (38 PA/team/game x 1.45 runs-per-wOBA-pt scaling.)
_RUNS_PER_WOBA_PT_PER_GAME = 38.0 * 1.45


ORDINAL_SUFFIXES = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = ORDINAL_SUFFIXES.get(n % 10, "th")
    return f"{n}{suffix}"


def _short_team(name: str) -> str:
    """Return a short tag for a team name (best-effort).

    Uses the last whitespace-separated token when the name has multiple
    words (e.g. "New York Yankees" → "Yankees"); otherwise uppercases.
    Falls back to the raw name when both heuristics return empty.
    """
    if not name:
        return ""
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1]
    return name


def _find_source(event: Event, name: str) -> SourceProb | None:
    for sp in event.source_probs:
        if sp.source == name:
            return sp
    return None


def _market_devig_home(event: Event) -> float | None:
    home_prices = [american_to_prob(ln.american) for ln in event.lines if ln.side == "home" and not ln.is_opening]
    away_prices = [american_to_prob(ln.american) for ln in event.lines if ln.side == "away" and not ln.is_opening]
    if not home_prices:
        home_prices = [american_to_prob(ln.american) for ln in event.lines if ln.side == "home"]
    if not away_prices:
        away_prices = [american_to_prob(ln.american) for ln in event.lines if ln.side == "away"]
    if not home_prices or not away_prices:
        return None
    h = sum(home_prices) / len(home_prices)
    a = sum(away_prices) / len(away_prices)
    h_devig, _ = devig_two_way(h, a)
    return h_devig


def _parse_kv(notes: str) -> dict[str, str]:
    """Pull key=value pairs out of a SourceProb.notes string."""
    out: dict[str, str] = {}
    for tok in (notes or "").split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def _generic_statcast_summary(event: Event, sp: SourceProb) -> str | None:
    """Original team-level lineup-edge string (fallback when per-batter data missing)."""
    kv = _parse_kv(sp.notes)
    try:
        diff = float(kv.get("diff", "0"))
    except ValueError:
        diff = 0.0
    direction = "home" if diff > 0 else "away"
    if abs(diff) < 0.001:
        return f"Statcast lineup matchup: neutral (xwOBA differential {diff:+.4f})."
    side = (event.home if direction == "home" else event.away).upper()
    # diff is in xwOBA units squared — convert to a rough runs/game scale.
    runs_per_game = diff * 38.0 * 1.45  # rough PA/game × runs-per-wOBA-pt
    return (
        f"Statcast lineup edge: {side} offense projected {runs_per_game:+.2f} runs/game "
        f"on lineup × starter handedness matchup (xwOBA diff {diff:+.4f})."
    )


def _load_contributions(event: Event, sp: SourceProb) -> list[dict]:
    """Return per-batter contribution rows from metadata, falling back to the DB.

    Three layers:
      1. ``sp.metadata['lineup_contributions']`` — the live path; the
         connector emits this directly on the SourceProb.
      2. ``data/source_history.db`` — the persisted table populated by the
         same connector; used when the event was loaded from disk (cached
         pipeline) without metadata.
      3. empty list — the explainer falls back to the generic string.
    """
    rows: list[dict] = []
    md = getattr(sp, "metadata", None)
    if isinstance(md, dict):
        lc = md.get("lineup_contributions")
        if isinstance(lc, list):
            rows = [r for r in lc if isinstance(r, dict)]
    if rows:
        return rows
    # Best-effort DB load — keep the explainer pure on any I/O failure.
    try:
        from .sources.mlb_statcast_lineup import load_lineup_contributions

        rows = load_lineup_contributions(event.event_id)
    except Exception:
        rows = []
    return rows


def _batter_lines(
    event: Event,
    contributions: list[dict],
    *,
    threshold: float = BATTER_RATIONALE_DEVIATION_THRESHOLD,
    top_n: int = 3,
) -> list[str]:
    """Return up to ``top_n`` per-batter rationale strings.

    Only batters whose xwOBA deviates from league mean by more than
    ``threshold`` (in absolute value) are eligible — we don't fabricate
    specificity from noise.
    """
    eligible: list[tuple[float, dict]] = []
    for row in contributions:
        if not row.get("batter_name"):
            continue
        if not row.get("xwoba_observed", True):
            # Missing batter xwOBA — connector filled with league avg; do
            # not surface that as a "matchup-driving" batter.
            continue
        try:
            x = float(row.get("xwoba_vs_handedness"))
            avg = float(row.get("league_avg_xwoba"))
        except (TypeError, ValueError):
            continue
        dev = x - avg
        if abs(dev) < threshold:
            continue
        eligible.append((abs(dev), row))
    eligible.sort(key=lambda t: t[0], reverse=True)
    out: list[str] = []
    for _, row in eligible[:top_n]:
        out.append(_format_batter_line(event, row))
    return out


def _format_batter_line(event: Event, row: dict) -> str:
    name = str(row.get("batter_name") or "").strip()
    team_name = row.get("team") or ""
    team_tag = _short_team(team_name)
    pos = int(row.get("batting_order_position") or 0)
    order_str = _ordinal(pos) if pos else "?"
    hand = (row.get("vs_pitcher_hand") or "").upper()
    hand_str = "LHP" if hand == "L" else ("RHP" if hand == "R" else "")
    starter_str = f"vs {hand_str} starter" if hand_str else "vs starter"
    try:
        x = float(row.get("xwoba_vs_handedness"))
        avg = float(row.get("league_avg_xwoba"))
    except (TypeError, ValueError):
        x, avg = 0.0, 0.0
    dev = x - avg
    pa_w = float(row.get("pa_weight") or 0.0)
    # Runs/game above neutral attributable to this batter, assuming a
    # league-average opposing pitcher. PA-weighted so a leadoff +0.080
    # xwOBA outlier scores higher than a 9-hole +0.080 outlier.
    runs_per_game = pa_w * dev * _RUNS_PER_WOBA_PT_PER_GAME
    pieces = [f"Statcast: {name}"]
    if team_tag and order_str:
        pieces[-1] += f" ({team_tag}, {order_str} in order)"
    elif team_tag:
        pieces[-1] += f" ({team_tag})"
    line = f"{pieces[0]} {starter_str} — {x:.3f} xwOBA vs handedness, league avg {avg:.3f}"
    if abs(runs_per_game) >= 0.005:
        line += f" → {runs_per_game:+.2f} runs/game above neutral"
    return line


def _statcast_explanation(event: Event) -> str | None:
    """Legacy single-string entry point. Returns the first generated line.

    Kept for backwards compatibility with callers / tests that expect a
    single string. ``_statcast_lines`` is the preferred multi-row entry
    point used by ``explain_event``.
    """
    lines = _statcast_lines(event)
    return lines[0] if lines else None


def _statcast_lines(event: Event) -> list[str]:
    sp = _find_source(event, "mlb-statcast-lineup")
    if sp is None:
        return []
    contributions = _load_contributions(event, sp)
    batter_lines = _batter_lines(event, contributions) if contributions else []
    if batter_lines:
        return batter_lines
    fallback = _generic_statcast_summary(event, sp)
    return [fallback] if fallback else []


def _weather_explanation(event: Event) -> str | None:
    sp = _find_source(event, "mlb-weather")
    if sp is None:
        return None
    notes = sp.notes or ""
    kv = _parse_kv(notes)
    if "dome=True" in notes:
        # Domes get no explanatory weather row — the source still
        # contributes a neutral run-environment prediction.
        return None
    if "no park table entry" in notes or "runs_h" not in kv:
        # Unknown venue — weather source still emits a neutral prob, but
        # we have nothing meaningful to say about the run environment.
        return None
    try:
        runs_h = float(kv.get("runs_h", "0"))
        runs_a = float(kv.get("runs_a", "0"))
    except ValueError:
        return None
    temp = kv.get("temp")
    wind = kv.get("wind")
    direction = kv.get("dir")
    avg_runs = (runs_h + runs_a) / 2.0
    baseline = 4.5
    pct = (avg_runs / baseline - 1.0) * 100.0
    pieces = []
    if temp:
        pieces.append(f"{temp} air")
    if wind and direction:
        pieces.append(f"{wind} wind from {direction}°")
    head = ", ".join(pieces) if pieces else "park-adjusted conditions"
    return f"Weather: {head} → run environment {pct:+.1f}% vs league baseline."


def _epa_explanation(event: Event) -> str | None:
    sp = _find_source(event, "nfl-nflfastr-epa")
    if sp is None:
        return None
    kv = _parse_kv(sp.notes)
    try:
        diff = float(kv.get("pred_diff", "0"))
        h_off = float(kv.get("h_off", "0"))
        a_off = float(kv.get("a_off", "0"))
    except ValueError:
        return None
    side = (event.home if diff >= 0 else event.away).upper()
    return (
        f"EPA edge: {side} projected {abs(diff):.1f}-point favorite "
        f"(off EPA {h_off:+.3f} vs {a_off:+.3f}, def diff layered)."
    )


def _cpoe_explanation(event: Event) -> str | None:
    """Rationale snippet for the NFL Next Gen Stats CPOE source."""
    sp = _find_source(event, "nfl-nextgen-cpoe")
    if sp is None:
        return None
    kv = _parse_kv(sp.notes)
    try:
        cpoe_diff = float(kv.get("cpoe_diff", "0").replace("pp", ""))
        h_cpoe = float(kv.get("h_cpoe", "0"))
        a_cpoe = float(kv.get("a_cpoe", "0"))
    except ValueError:
        return None
    side = (event.home if cpoe_diff >= 0 else event.away).upper()
    return (
        f"QB CPOE edge: {side} +{abs(cpoe_diff):.1f}pp over expected "
        f"(home CPOE {h_cpoe:+.2f}, away CPOE {a_cpoe:+.2f})."
    )


def _srs_explanation(event: Event) -> str | None:
    sp = _find_source(event, "nba-bref-srs-pace")
    if sp is None:
        return None
    kv = _parse_kv(sp.notes)
    try:
        diff = float(kv.get("diff", "0"))
        h_srs = float(kv.get("h_srs", "0"))
        a_srs = float(kv.get("a_srs", "0"))
    except ValueError:
        return None
    side = (event.home if diff >= 0 else event.away).upper()
    return (
        f"SRS edge: {side} favored by {abs(diff):.1f} points "
        f"(home SRS {h_srs:+.2f}, away SRS {a_srs:+.2f}, +2.5 HFA)."
    )


def _market_consensus_explanation(event: Event) -> str | None:
    market = _market_devig_home(event)
    if market is None or event.blended_home_prob is None:
        return None
    side = event.pick
    if side is None:
        return None
    market_side = market if side == "home" else (1.0 - market)
    blended_side = event.blended_home_prob if side == "home" else (1.0 - event.blended_home_prob)
    edge_pp = (blended_side - market_side) * 100.0
    return (
        f"Market consensus {market_side:.0%}, blended model {blended_side:.0%} — "
        f"{edge_pp:+.1f}pp edge on the {side.upper()} side."
    )


def _signal_explanation(event: Event) -> str | None:
    if not event.signals:
        return None
    pretty = []
    for s in event.signals:
        if s == "chalk-overpriced":
            pretty.append("market overpaying for chalk")
        elif s.startswith("reverse-line-movement"):
            pretty.append("reverse line movement")
        elif s == "book-dispersion-dog":
            pretty.append("book dispersion on the dog")
    if not pretty:
        return None
    return "Active signals: " + ", ".join(pretty) + "."


# Priority order for the explanation list. Each callable returns a string
# or None; we keep the non-None results in order and stop at top-N.
PRIORITY_FNS = (
    _statcast_explanation,
    _weather_explanation,
    _epa_explanation,
    _cpoe_explanation,
    _srs_explanation,
    _market_consensus_explanation,
    _signal_explanation,
)


def explain_event(event: Event, *, top_n: int = 3) -> list[str]:
    """Return ordered list of plain-text rationale strings (up to ``top_n``).

    For MLB events the Statcast block can expand into multiple per-batter
    lines when per-batter contribution data is available and at least one
    batter clears the deviation threshold. Other priority slots remain
    single-string — the order below puts batter lines first, then weather,
    market consensus, etc.
    """
    out: list[str] = []

    # Statcast block first: 0..N lines depending on per-batter data.
    if event.sport == "mlb":
        for s in _statcast_lines(event):
            if len(out) >= top_n:
                break
            out.append(s)

    for fn in PRIORITY_FNS:
        if fn is _statcast_explanation:
            # Already handled above with the multi-line path.
            continue
        if len(out) >= top_n:
            break
        s = fn(event)
        if s:
            out.append(s)
    return out
