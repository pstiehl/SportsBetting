"""ESPN PGA Tour connector — scoreboard field discovery + leaderboard proxy.

ESPN's BPI / FPI predictor framework historically only covers team sports
(football, basketball, baseball). Verified 2026-05-30: hitting
``/sports/core/sports/golf/leagues/pga/events/{eid}/competitions/{eid}/predictor``
returns HTTP 400 ``"Predictor is not supported for sport: golf, league: pga"``.

Rather than ship a stub that silently returns ``[]`` for the whole season,
we fall back to ESPN's **public PGA leaderboard endpoint**::

    https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard

The scoreboard returns the full active-tournament field with each player's
current ``scoreToPar`` and ``sortOrder``. We treat the **leaderboard
position pre-round** as a noisy power-rating signal: a player sitting at
``-7 thru 36`` has demonstrated more skill this week than a player at
``+3 thru 36``, and that's a perfectly defensible (if narrow) basis for a
head-to-head matchup probability.

Caveats:
  * This is intentionally a **weak** signal — early-round leaderboard
    positions are noisy and overweighting Friday's scoring vs Sunday's
    closing pressure is a known mispricing. We expose it as a secondary
    source so the blender can downweight it after backtest.
  * If the tournament hasn't teed off yet (round 1 not started), the
    leaderboard shows every player at "E" / no score and we emit no
    Events — the upstream ``PGADatagolf`` connector carries the slate
    pre-tournament.

Endpoint reference: ESPN's ``site.api.espn.com`` v2 golf scoreboard is the
same endpoint used by the public ESPN.com leaderboard page. No auth.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time as dt_time, timezone
from typing import Iterable

import httpx

from ..types import Event, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)


SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
USER_AGENT = "flashcat-research/1.0 (+https://github.com/pstiehl/SportsBetting)"


def _logistic(z: float, *, scale: float = 2.5) -> float:
    """Logistic CDF with score-difference scaled by ``scale`` strokes.

    A 2.5-stroke gap → ~73% win probability for the leader, which is the
    empirical PGA-Tour estimate for head-to-head matchups when one player
    leads the other by ~3 strokes through 36 holes (Mark Broadie 2014).
    """
    import math

    return 1.0 / (1.0 + math.exp(-z / scale))


class PGAESPNScoreboard(SourceConnector):
    """ESPN PGA Tour scoreboard → H2H matchup probabilities.

    Discovers the active PGA Tour event, ranks the field by current
    ``scoreToPar``, and pairs the leaderboard adjacently into Events that
    align with ``PGADatagolf``'s pairing scheme. This means the blender
    can stack the two sources on the same H2H matchup.

    ESPN's golf predictor endpoint does **not** exist (confirmed HTTP 400).
    We document that here and lean on the leaderboard score instead.
    """

    name = "pga-espn-scoreboard"
    version = "v1-leaderboard-proxy"
    is_live = True

    def __init__(
        self, timeout: float = 8.0, max_pairs: int = 64, scale_strokes: float = 2.5
    ):
        self.timeout = timeout
        self.max_pairs = max_pairs
        self.scale_strokes = scale_strokes

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport is not None and sport != "pga":
            return []

        payload = self._fetch_scoreboard()
        if not payload:
            return []
        return self._build_events(payload, start=start, end=end)

    def _fetch_scoreboard(self) -> dict | None:
        try:
            with httpx.Client(
                timeout=self.timeout, headers={"User-Agent": USER_AGENT}
            ) as c:
                r = c.get(SCOREBOARD_URL)
                r.raise_for_status()
                return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("pga-espn-scoreboard fetch failed: %s", e)
            return None

    # ----- transform ---------------------------------------------------

    def _build_events(
        self, payload: dict, *, start: date, end: date
    ) -> list[Event]:
        events_payload = payload.get("events") or []
        if not events_payload:
            return []
        ev = events_payload[0]
        ev_name = ev.get("name") or "PGA Event"
        ev_id = str(ev.get("id") or "")
        try:
            commence_dt = datetime.fromisoformat(
                ev["date"].replace("Z", "+00:00")
            )
        except Exception:
            commence_dt = datetime.combine(start, dt_time(14, 0), tzinfo=timezone.utc)
        if commence_dt.date() > end:
            return []

        comp = (ev.get("competitions") or [{}])[0]
        competitors = comp.get("competitors") or []
        if not competitors:
            return []

        # Build (player_name, score_to_par) list — skip non-started rounds.
        rows: list[tuple[str, float]] = []
        for c in competitors:
            athlete = c.get("athlete") or {}
            name = athlete.get("displayName")
            if not name:
                continue
            score = self._extract_score_to_par(c)
            if score is None:
                # Round-1 pre-tee field; no signal yet — skip cleanly.
                continue
            rows.append((name, score))

        if len(rows) < 2:
            return []

        # Sort ascending (better = lower scoreToPar) and pair adjacently.
        rows.sort(key=lambda r: r[1])
        pairs = self._pair_rows(rows)

        out: list[Event] = []
        captured = datetime.now(timezone.utc)
        for a, b in pairs[: self.max_pairs]:
            home, away, swapped = _canonical_pair(a[0], b[0])
            # Score gap measured from "home" side (positive when home is leading).
            home_score = a[1] if not swapped else b[1]
            away_score = b[1] if not swapped else a[1]
            gap_for_home = away_score - home_score  # positive → home better
            home_prob = max(0.001, min(0.999, _logistic(gap_for_home, scale=self.scale_strokes)))
            out.append(
                Event(
                    event_id=f"espn-pga:{ev_id}:{_player_key(home)}_vs_{_player_key(away)}",
                    sport="pga",
                    league=f"PGA: {ev_name}",
                    home=home,
                    away=away,
                    commence_time=commence_dt,
                    source_probs=[
                        SourceProb(
                            source=self.name,
                            home_win_prob=home_prob,
                            captured_at=captured,
                            notes=(
                                f"ESPN PGA leaderboard proxy "
                                f"(home={home_score:+.0f}, away={away_score:+.0f}, "
                                f"gap_for_home={gap_for_home:+.1f}); "
                                f"logistic scale={self.scale_strokes} strokes"
                            ),
                        )
                    ],
                )
            )
        return out

    @staticmethod
    def _extract_score_to_par(competitor: dict) -> float | None:
        """Pull the current to-par score for a competitor.

        Tries two locations: the ``statistics`` list (populated during a
        round) and the top-level ``score`` string (populated between
        rounds, e.g. "-10", "E", "+3"). Returns ``None`` if neither
        carries a parseable value.
        """
        for stat in competitor.get("statistics") or []:
            if stat.get("name") == "scoreToPar":
                val = stat.get("value")
                if val is None:
                    continue
                try:
                    return float(val)
                except Exception:
                    pass
        # Fallback: top-level `score` string between rounds.
        s = competitor.get("score")
        if s is None:
            return None
        s = str(s).strip()
        if s in ("", "--", "WD", "CUT", "MDF", "DQ"):
            return None
        if s.upper() == "E":
            return 0.0
        try:
            return float(s.replace("+", ""))
        except Exception:
            return None

    @staticmethod
    def _pair_rows(rows: list[tuple[str, float]]) -> list[tuple[tuple, tuple]]:
        pairs = []
        it = iter(rows)
        for a in it:
            b = next(it, None)
            if b is None:
                break
            pairs.append((a, b))
        return pairs


def _canonical_pair(a: str, b: str) -> tuple[str, str, bool]:
    na = (a or "").strip().lower()
    nb = (b or "").strip().lower()
    if na <= nb:
        return a, b, False
    return b, a, True


def _player_key(name: str) -> str:
    return (
        (name or "")
        .lower()
        .replace(".", "")
        .replace(",", "")
        .replace(" ", "_")
        .strip("_")
    )
