"""Per-pick "Why this pick" rationale generator.

For each event we surface the top-3 signals driving the model's pick. The
rendered event card embeds these inside a ``<details>`` block on the live
site so Phil can audit the decision without parsing source-by-source
probabilities.

Priority order (highest leverage first):
  1. Statcast lineup edge (MLB only) — biggest single-game signal.
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


def _statcast_explanation(event: Event) -> str | None:
    sp = _find_source(event, "mlb-statcast-lineup")
    if sp is None:
        return None
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
    _srs_explanation,
    _market_consensus_explanation,
    _signal_explanation,
)


def explain_event(event: Event, *, top_n: int = 3) -> list[str]:
    """Return ordered list of plain-text rationale strings (up to ``top_n``)."""
    out: list[str] = []
    for fn in PRIORITY_FNS:
        if len(out) >= top_n:
            break
        s = fn(event)
        if s:
            out.append(s)
    return out
