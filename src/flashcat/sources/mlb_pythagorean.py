"""MLB Pythagorean expectation — a computed source.

For each game, we compute a Pythagorean win expectancy from each team's
season-to-date runs-scored / runs-allowed:

    W%_team = RS^γ / (RS^γ + RA^γ)            (Bill James, γ ≈ 1.83)

Per-game home prob is then a logistic-Bradley-Terry of the two W% values
plus a 4 percentage-point home-field bump (the standard MLB HFA).

We reuse the 538 MLB Elo archive as the source of game-level
``score1`` / ``score2`` data because:
  1. it covers 1871-2023 with one CSV,
  2. games are already keyed by canonical team codes, and
  3. it's already cached in CI.

Walk-forward strictness: only games strictly BEFORE the current game's date
contribute to season-to-date totals. The first ~30 games of any team's
season fall back to the league-average regression value (4.5 RPG with γ=1.83
gives 50/50).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, datetime, time, timezone

from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector
from .fivethirtyeight_archives import (
    MLB_URL,
    _download_csv,
    _safe_float,
)


def _normalize_team(name: str) -> str:
    return (name or "").lower().replace(".", "").strip()

log = logging.getLogger(__name__)

GAMMA = 1.83
HFA_PP = 0.04  # 4 pp home-field advantage
MIN_GAMES = 20  # below this we regress hard toward .500


def _pyth_win_pct(rs: float, ra: float) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    rs_g = max(0.01, rs) ** GAMMA
    ra_g = max(0.01, ra) ** GAMMA
    return rs_g / (rs_g + ra_g)


def _bt_prob(home_pct: float, away_pct: float) -> float:
    """Bradley-Terry: P(A > B) = pA / (pA + pB) on win odds."""
    # Convert to odds-like
    oa = home_pct / max(1e-6, 1 - home_pct)
    ob = away_pct / max(1e-6, 1 - away_pct)
    return oa / (oa + ob)


class MLBPythagorean(SourceConnector):
    name = "mlb-pythagorean"
    version = "james-1.83"
    is_live = True

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        events, _ = self._compute(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._compute(start, end)
        return results

    def _compute(self, start: date, end: date) -> tuple[list[Event], list[HistoricalResult]]:
        rows = _download_csv(MLB_URL, "538_mlb_elo.csv", timeout=self.timeout)
        # Sort by date so walk-forward is correct.
        parsed: list[dict] = []
        for r in rows:
            try:
                d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            s1 = _safe_float(r.get("score1"))
            s2 = _safe_float(r.get("score2"))
            if s1 is None or s2 is None:
                continue
            parsed.append(
                {
                    "date": d,
                    "season": r.get("season") or str(d.year),
                    "team1": r.get("team1") or "",
                    "team2": r.get("team2") or "",
                    "score1": s1,
                    "score2": s2,
                }
            )
        parsed.sort(key=lambda x: x["date"])

        # Per-season running totals.
        # season -> team -> {"rs": ..., "ra": ..., "g": ...}
        totals: dict[tuple, dict[str, float]] = defaultdict(
            lambda: {"rs": 0.0, "ra": 0.0, "g": 0.0}
        )

        events: list[Event] = []
        results: list[HistoricalResult] = []

        for row in parsed:
            d = row["date"]
            season = row["season"]
            t1 = row["team1"]
            t2 = row["team2"]

            # Prediction BEFORE recording today's game.
            t1_stats = totals[(season, t1)]
            t2_stats = totals[(season, t2)]
            if t1_stats["g"] < MIN_GAMES or t2_stats["g"] < MIN_GAMES:
                # Regress toward .500 with shrinkage weight.
                w1 = min(1.0, t1_stats["g"] / MIN_GAMES)
                w2 = min(1.0, t2_stats["g"] / MIN_GAMES)
                pct1 = 0.5 + w1 * (_pyth_win_pct(t1_stats["rs"] / max(1, t1_stats["g"]),
                                                  t1_stats["ra"] / max(1, t1_stats["g"])) - 0.5)
                pct2 = 0.5 + w2 * (_pyth_win_pct(t2_stats["rs"] / max(1, t2_stats["g"]),
                                                  t2_stats["ra"] / max(1, t2_stats["g"])) - 0.5)
            else:
                pct1 = _pyth_win_pct(t1_stats["rs"] / t1_stats["g"],
                                     t1_stats["ra"] / t1_stats["g"])
                pct2 = _pyth_win_pct(t2_stats["rs"] / t2_stats["g"],
                                     t2_stats["ra"] / t2_stats["g"])
            # team1 is "home" in 538's schema. Apply HFA.
            base = _bt_prob(pct1, pct2)
            home_prob = max(0.05, min(0.95, base + HFA_PP))

            if start <= d <= end:
                home_won = row["score1"] > row["score2"]
                commence = datetime.combine(d, time(19, 0), tzinfo=timezone.utc)
                event_id = (
                    f"mlbpyth:{season}_{d.isoformat()}_"
                    f"{_normalize_team(t1)}_{_normalize_team(t2)}"
                )
                events.append(
                    Event(
                        event_id=event_id,
                        sport="mlb",
                        league="MLB",
                        home=t1,
                        away=t2,
                        commence_time=commence,
                        source_probs=[
                            SourceProb(
                                source="mlb-pythagorean",
                                home_win_prob=home_prob,
                                captured_at=commence,
                                notes=(
                                    f"Bill James pythag γ={GAMMA}, "
                                    f"home {t1_stats['g']:.0f}g rs={t1_stats['rs']:.0f}, "
                                    f"away {t2_stats['g']:.0f}g rs={t2_stats['rs']:.0f}"
                                ),
                            )
                        ],
                    )
                )
                results.append(
                    HistoricalResult(
                        event_id=event_id,
                        sport="mlb",
                        home=t1,
                        away=t2,
                        commence_time=commence,
                        home_won=home_won,
                        home_score=int(row["score1"]),
                        away_score=int(row["score2"]),
                    )
                )

            # Update totals after the prediction (true walk-forward).
            t1_stats["rs"] += row["score1"]
            t1_stats["ra"] += row["score2"]
            t1_stats["g"] += 1
            t2_stats["rs"] += row["score2"]
            t2_stats["ra"] += row["score1"]
            t2_stats["g"] += 1

        return events, results
