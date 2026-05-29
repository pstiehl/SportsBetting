"""ESPN public scoreboard JSON connector.

URL: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard
No auth. Returns events, teams, and (sometimes) a pre-game win probability
under competitions[].competitors[].records or via the predictor endpoint.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import httpx

from ..config import CACHE_DIR
from ..types import Event, Sport, SourceProb
from .base import SourceConnector

log = logging.getLogger(__name__)

ENDPOINTS: dict[Sport, tuple[str, str]] = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "cfb": ("football", "college-football"),
    "cbb": ("basketball", "mens-college-basketball"),
}


class ESPNScoreboard(SourceConnector):
    name = "espn-scoreboard"
    version = "site-v2"
    is_live = True

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        sports = [sport] if sport else list(ENDPOINTS.keys())
        out: list[Event] = []
        for s in sports:
            try:
                out.extend(self._fetch_sport(s, start))
            except Exception as e:  # noqa: BLE001
                log.warning("espn fetch failed for %s: %s", s, e)
        return out

    def _fetch_sport(self, sport: Sport, day: date) -> list[Event]:
        sport_grp, league = ENDPOINTS[sport]
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_grp}/{league}/scoreboard"
        )
        params = {"dates": day.strftime("%Y%m%d")}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("espn fetch failed: %s", e)
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"espn_{sport}_{day}.json", "w") as f:
            json.dump(data, f, indent=2)
        return self._parse(data, sport)

    @staticmethod
    def _parse(data: dict, sport: Sport) -> list[Event]:
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        for ev in data.get("events", []):
            event_id = f"espn:{ev.get('id')}"
            try:
                commence = datetime.fromisoformat(
                    ev["date"].replace("Z", "+00:00")
                )
            except Exception:
                commence = now
            home_name = away_name = ""
            home_prob = None
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            for c in comp.get("competitors", []):
                team_name = c.get("team", {}).get("displayName", "")
                if c.get("homeAway") == "home":
                    home_name = team_name
                else:
                    away_name = team_name
            # Pre-game win probability lives in comp["predictor"] when ESPN provides it.
            predictor = comp.get("predictor") or {}
            home_team_p = (predictor.get("homeTeam") or {}).get("gameProjection")
            if home_team_p is not None:
                try:
                    home_prob = float(home_team_p) / 100.0
                except Exception:
                    home_prob = None
            source_probs: list[SourceProb] = []
            if home_prob is not None:
                source_probs.append(
                    SourceProb(
                        source="espn-scoreboard",
                        home_win_prob=max(0.001, min(0.999, home_prob)),
                        captured_at=now,
                        notes="ESPN predictor gameProjection",
                    )
                )
            out.append(
                Event(
                    event_id=event_id,
                    sport=sport,
                    league=sport.upper(),
                    home=home_name,
                    away=away_name,
                    commence_time=commence,
                    source_probs=source_probs,
                )
            )
        return out
