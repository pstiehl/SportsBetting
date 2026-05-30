"""ESPN FPI (Football Power Index) predictor for CFB.

Mirror of the NFL leg of ``espn_predictor.ESPNPredictor`` for college football.
The ESPN core API exposes the same predictor schema as the NFL endpoint at::

    https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/
       events/{eventId}/competitions/{eventId}/predictor

The home team's ``gameProjection`` field is the FPI-derived home-win
probability (in percent). We divide by 100 and emit it as a SourceProb
under the ``espn-fpi-cfb`` source name so the per-sport accuracy weighter
can track CFB's predictor independently from NFL's.

This is a **live / forward-only** source. ESPN doesn't archive historical
predictor snapshots, so the connector contributes nothing to the
historical backtest path (it returns ``[]`` for old dates).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx

from ..types import Event, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

_USER_AGENT = "flashcat-research/1.0"


def _scoreboard_url(day: date) -> str:
    return (
        "https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
        f"scoreboard?dates={day.strftime('%Y%m%d')}"
    )


def _predictor_url(eid: str) -> str:
    return (
        "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
        f"events/{eid}/competitions/{eid}/predictor"
    )


def _extract_game_projection(predictor: dict) -> float | None:
    """Find the homeTeam ``gameProjection`` (0-100 → 0-1) in the predictor payload."""
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


class CFBESPNFPI(SourceConnector):
    """ESPN FPI live home-win probabilities for CFB."""

    name = "espn-fpi-cfb"
    version = "core-v2"
    is_live = True

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "cfb":
            return []
        out: list[Event] = []
        cur = start
        while cur <= end:
            try:
                out.extend(self._fetch_day(cur))
            except Exception as e:  # noqa: BLE001
                log.debug("espn-fpi-cfb %s failed: %s", cur, e)
            cur = date.fromordinal(cur.toordinal() + 1)
        return out

    def _fetch_day(self, day: date) -> list[Event]:
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        url = _scoreboard_url(day)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("ESPN CFB scoreboard fetch failed for %s: %s", day, e)
            return []
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        with httpx.Client(timeout=self.timeout) as client:
            for ev in data.get("events", []) or []:
                eid = ev.get("id")
                if not eid:
                    continue
                try:
                    commence = datetime.fromisoformat(
                        str(ev.get("date") or "").replace("Z", "+00:00")
                    )
                except Exception:
                    commence = now
                home_name = away_name = ""
                comps = ev.get("competitions") or []
                if not comps:
                    continue
                for c in comps[0].get("competitors", []) or []:
                    name = (c.get("team") or {}).get("displayName", "")
                    if c.get("homeAway") == "home":
                        home_name = name
                    elif c.get("homeAway") == "away":
                        away_name = name
                if not home_name or not away_name:
                    continue
                try:
                    pr = client.get(_predictor_url(eid), headers=headers)
                    pr.raise_for_status()
                    predictor = pr.json()
                except Exception as e:  # noqa: BLE001
                    log.debug("espn-fpi-cfb predictor %s failed: %s", eid, e)
                    continue
                home_prob = _extract_game_projection(predictor)
                if home_prob is None:
                    continue
                out.append(Event(
                    event_id=f"espn-fpi-cfb:{eid}",
                    sport="cfb",
                    league="NCAAF",
                    home=home_name,
                    away=away_name,
                    commence_time=commence,
                    source_probs=[
                        SourceProb(
                            source=self.name,
                            home_win_prob=max(0.001, min(0.999, home_prob)),
                            captured_at=now,
                            notes="ESPN core FPI predictor.gameProjection",
                        )
                    ],
                ))
        return out
