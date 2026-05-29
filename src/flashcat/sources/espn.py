"""ESPN public scoreboard JSON connector.

URL: https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard
No auth. Returns events, teams, and (sometimes) a pre-game win probability
under ``competitions[].predictor.homeTeam.gameProjection``.

Tennis is handled specially: ESPN returns a tournament object whose
``groupings[].competitions[]`` list contains the individual matches (with
``athletes`` instead of ``teams``). Matches are filtered to the requested
date range and yielded as Events.
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

# Team-sport endpoints: ESPN expects ?dates=YYYYMMDD for a single-day scoreboard.
TEAM_ENDPOINTS: dict[Sport, tuple[str, str]] = {
    "nfl": ("football", "nfl"),
    "nba": ("basketball", "nba"),
    "mlb": ("baseball", "mlb"),
    "nhl": ("hockey", "nhl"),
    "cfb": ("football", "college-football"),
    "cbb": ("basketball", "mens-college-basketball"),
}

# Tennis endpoints: per-tour scoreboard. No date param — returns the current
# tournament with its full draw; filter client-side.
TENNIS_ENDPOINTS: dict[Sport, tuple[str, str]] = {
    "atp": ("tennis", "atp"),
    "wta": ("tennis", "wta"),
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
        if sport is not None:
            sports: list[Sport] = [sport]
        else:
            sports = list(TEAM_ENDPOINTS.keys()) + list(TENNIS_ENDPOINTS.keys())
        out: list[Event] = []
        for s in sports:
            try:
                if s in TENNIS_ENDPOINTS:
                    out.extend(self._fetch_tennis(s, start, end))
                else:
                    # Iterate per-day in the requested range (usually 1-3 days).
                    cur = start
                    while cur <= end:
                        out.extend(self._fetch_team_sport(s, cur))
                        cur = cur.fromordinal(cur.toordinal() + 1)
            except Exception as e:  # noqa: BLE001
                log.warning("espn fetch failed for %s: %s", s, e)
        return out

    # ----- team sports -------------------------------------------------

    def _fetch_team_sport(self, sport: Sport, day: date) -> list[Event]:
        sport_grp, league = TEAM_ENDPOINTS[sport]
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
        return self._parse_team(data, sport, day)

    @staticmethod
    def _parse_team(data: dict, sport: Sport, request_day: date) -> list[Event]:
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
            # Belt-and-braces: ESPN sometimes returns events from neighbouring
            # days when a date param isn't strict. Drop anything more than ±1
            # day from the requested date.
            delta_days = abs((commence.date() - request_day).days)
            if delta_days > 1:
                continue
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

    # ----- tennis ------------------------------------------------------

    def _fetch_tennis(self, sport: Sport, start: date, end: date) -> list[Event]:
        sport_grp, league = TENNIS_ENDPOINTS[sport]
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/"
            f"{sport_grp}/{league}/scoreboard"
        )
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("espn tennis fetch failed: %s", e)
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"espn_{sport}.json", "w") as f:
            json.dump(data, f, indent=2)
        return self._parse_tennis(data, sport, start, end)

    @staticmethod
    def _parse_tennis(
        data: dict, sport: Sport, start: date, end: date
    ) -> list[Event]:
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        for tournament in data.get("events", []):
            tname = tournament.get("shortName") or tournament.get("name") or sport.upper()
            for grouping in tournament.get("groupings", []):
                # Only keep singles. Doubles, mixed-doubles and qualifying
                # generally lack reliable moneylines and are dropped.
                gname = (grouping.get("grouping") or {}).get("slug") or ""
                if "double" in gname.lower():
                    continue
                for comp in grouping.get("competitions", []):
                    try:
                        commence = datetime.fromisoformat(
                            (comp.get("date") or "").replace("Z", "+00:00")
                        )
                    except Exception:
                        continue
                    if not (start <= commence.date() <= end):
                        continue
                    # ESPN only ever exposes singles inside the tournament
                    # tree — there's no concept of home/away for tennis, so
                    # we treat the first listed competitor as "home" by
                    # convention. This matches the OddsAPI convention.
                    competitors = comp.get("competitors", [])
                    if len(competitors) != 2:
                        continue
                    p1 = competitors[0].get("athlete", {}).get("displayName", "")
                    p2 = competitors[1].get("athlete", {}).get("displayName", "")
                    if not p1 or not p2:
                        continue
                    event_id = f"espn:{sport}:{comp.get('id')}"
                    out.append(
                        Event(
                            event_id=event_id,
                            sport=sport,
                            league=tname,
                            home=p1,
                            away=p2,
                            commence_time=commence,
                            source_probs=[],
                        )
                    )
        return out
