"""ESPN BPI/FPI/SOR predictor for MLB / NBA / NFL.

Different from ``ESPNScoreboard`` in two ways:

1. We hit the **core API predictor endpoint** directly for each upcoming game,
   which returns the full BPI/FPI projection (``gameProjection``,
   ``matchupQuality``, ``starterAdjustment`` for MLB, etc.).
2. We surface a SEPARATE source per sport: ``espn-bpi-mlb``,
   ``espn-bpi-nba``, ``espn-fpi-nfl``. This lets the per-sport accuracy
   weighter track each sport's predictor independently — the previous
   blanket ``espn-scoreboard`` source pooled them, which dragged its
   per-sport weight around.

This is a **live / forward-only** source. We never claim historical backtest
coverage from it — ESPN doesn't expose archived predictor snapshots, and
re-running the predictor today against historical games would be a
post-hoc leak. The backtest harness skips this connector cleanly.

Endpoint pattern:
    https://sports.core.api.espn.com/v2/sports/{group}/leagues/{league}/events/{eid}/competitions/{eid}/predictor
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Iterable

import httpx

from ..types import Event, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

_SPORT_TO_GROUP: dict[str, tuple[str, str, str]] = {
    "mlb": ("baseball", "mlb", "espn-bpi-mlb"),
    "nba": ("basketball", "nba", "espn-bpi-nba"),
    "nfl": ("football", "nfl", "espn-fpi-nfl"),
}


def _scoreboard_url(group: str, league: str, day: date) -> str:
    return (
        f"https://site.api.espn.com/apis/site/v2/sports/{group}/{league}/scoreboard"
        f"?dates={day.strftime('%Y%m%d')}"
    )


def _predictor_url(group: str, league: str, eid: str) -> str:
    return (
        f"https://sports.core.api.espn.com/v2/sports/{group}/leagues/{league}/"
        f"events/{eid}/competitions/{eid}/predictor"
    )


def _extract_game_projection(predictor: dict) -> float | None:
    """Find the homeTeam ``gameProjection`` value in the predictor payload."""
    home = predictor.get("homeTeam") or {}
    for stat in home.get("statistics") or []:
        if stat.get("name") == "gameProjection":
            val = stat.get("value")
            if val is None:
                continue
            try:
                return float(val) / 100.0
            except Exception:
                return None
    return None


class ESPNPredictor(SourceConnector):
    """Live BPI/FPI predictor for MLB / NBA / NFL."""

    name = "espn-predictor"
    version = "core-v2"
    is_live = True

    def __init__(self, sports: Iterable[str] = ("mlb", "nba", "nfl"), timeout: float = 8.0):
        self.sports = list(sports)
        self.timeout = timeout

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        sports = [s for s in self.sports if (sport is None or s == sport)]
        out: list[Event] = []
        for s in sports:
            if s not in _SPORT_TO_GROUP:
                continue
            group, league, source_name = _SPORT_TO_GROUP[s]
            cur = start
            while cur <= end:
                try:
                    out.extend(self._fetch_day(s, group, league, source_name, cur))
                except Exception as e:  # noqa: BLE001
                    log.warning("espn-predictor %s %s failed: %s", s, cur, e)
                cur = date.fromordinal(cur.toordinal() + 1)
        return out

    def _fetch_day(
        self,
        sport: Sport,
        group: str,
        league: str,
        source_name: str,
        day: date,
    ) -> list[Event]:
        url = _scoreboard_url(group, league, day)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("espn-predictor scoreboard fetch failed: %s", e)
            return []
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        with httpx.Client(timeout=self.timeout) as client:
            for ev in data.get("events", []):
                eid = ev.get("id")
                if not eid:
                    continue
                try:
                    commence = datetime.fromisoformat(
                        ev["date"].replace("Z", "+00:00")
                    )
                except Exception:
                    commence = now
                home_name = away_name = ""
                comps = ev.get("competitions") or []
                if not comps:
                    continue
                for c in comps[0].get("competitors", []):
                    team_name = c.get("team", {}).get("displayName", "")
                    if c.get("homeAway") == "home":
                        home_name = team_name
                    else:
                        away_name = team_name
                if not home_name or not away_name:
                    continue
                # Fetch the predictor — best-effort.
                try:
                    pr = client.get(_predictor_url(group, league, eid))
                    pr.raise_for_status()
                    predictor = pr.json()
                except Exception as e:  # noqa: BLE001
                    log.debug("espn-predictor predictor %s failed: %s", eid, e)
                    continue
                home_prob = _extract_game_projection(predictor)
                if home_prob is None:
                    continue
                event_id = f"espn:{eid}"
                out.append(
                    Event(
                        event_id=event_id,
                        sport=sport,
                        league=sport.upper(),
                        home=home_name,
                        away=away_name,
                        commence_time=commence,
                        source_probs=[
                            SourceProb(
                                source=source_name,
                                home_win_prob=max(0.001, min(0.999, home_prob)),
                                captured_at=now,
                                notes="ESPN core BPI/FPI predictor.gameProjection",
                            )
                        ],
                    )
                )
        return out
