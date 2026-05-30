"""CFB market consensus connector.

Two-tier strategy mirroring the existing market sources:

  1. If ``THE_ODDS_API_KEY`` is configured (and the upstream ``TheOddsAPI``
     connector is wired in via ``flashcat.cli``), market lines flow in
     through that path with sport key ``americanfootball_ncaaf``. This
     connector then *augments* the slate with ESPN's published moneylines
     for CFB games (free + un-keyed) so we have coverage even on weeks
     where the Odds API doesn't return every matchup.

  2. If no Odds API key is available, this connector becomes the *primary*
     market line source for CFB. It pulls ESPN's CFB scoreboard endpoint
     and extracts the consensus moneyline + spread (ESPN publishes the
     "competition.odds" array on every event with at least one provider's
     prices, typically Bet365 / ESPN BET).

ESPN endpoint:
    https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates=YYYYMMDD

We emit ``BookLine`` rows tagged ``book="espn-consensus"`` so the meta-model's
market-close synthesizer can devig and slot it next to Bovada / FanDuel
prices for the other sports.

No API key required. Rate-friendly — ESPN's scoreboard endpoint serves a
whole day of games in one call.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx

from ..types import BookLine, Event, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/"
    "scoreboard?dates={ymd}"
)

_USER_AGENT = "flashcat-research/1.0"


def _parse_american(value) -> int | None:
    """Coerce ESPN ``moneyLine`` (sometimes string, sometimes int) to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            s = str(value).strip().replace("+", "")
            return int(float(s))
        except Exception:
            return None


def _extract_moneylines(odds: list[dict]) -> dict[str, int]:
    """Return ``{"home": -150, "away": +130}`` from ESPN's odds payload.

    ESPN's ``competition.odds[0]`` typically has ``homeTeamOdds.moneyLine``
    and ``awayTeamOdds.moneyLine``. Sometimes only ``details`` (the spread
    string) is populated; in that case we return ``{}`` and let the
    downstream blender skip.
    """
    out: dict[str, int] = {}
    if not odds:
        return out
    entry = odds[0]
    h_odds = (entry.get("homeTeamOdds") or {})
    a_odds = (entry.get("awayTeamOdds") or {})
    h_ml = _parse_american(h_odds.get("moneyLine"))
    a_ml = _parse_american(a_odds.get("moneyLine"))
    if h_ml is not None:
        out["home"] = h_ml
    if a_ml is not None:
        out["away"] = a_ml
    return out


class CFBMarketConsensus(SourceConnector):
    """ESPN CFB scoreboard market lines (no API key required)."""

    name = "cfb-market-consensus"
    version = "espn-1.0"
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
                log.debug("cfb-market-consensus %s failed: %s", cur, e)
            cur = date.fromordinal(cur.toordinal() + 1)
        return out

    def _fetch_day(self, day: date) -> list[Event]:
        url = _SCOREBOARD_URL.format(ymd=day.strftime("%Y%m%d"))
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("ESPN CFB scoreboard fetch failed for %s: %s", day, e)
            return []
        events: list[Event] = []
        now = datetime.now(timezone.utc)
        for ev in data.get("events", []) or []:
            eid = ev.get("id")
            if not eid:
                continue
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors") or []
            home = away = ""
            for c in competitors:
                team = (c.get("team") or {}).get("displayName") or ""
                if c.get("homeAway") == "home":
                    home = team
                elif c.get("homeAway") == "away":
                    away = team
            if not home or not away:
                continue
            try:
                commence = datetime.fromisoformat(
                    str(ev.get("date") or "").replace("Z", "+00:00")
                )
            except Exception:
                commence = now
            ml = _extract_moneylines(comp.get("odds") or [])
            lines: list[BookLine] = []
            for side, american in ml.items():
                lines.append(BookLine(
                    book="espn-consensus",
                    side=side,  # type: ignore[arg-type]
                    american=int(american),
                    captured_at=now,
                ))
            # Even without moneylines we still emit the event with no lines
            # so other connectors can merge against it on team-pair key.
            events.append(Event(
                event_id=f"cfb-market-consensus:{eid}",
                sport="cfb",
                league="NCAAF",
                home=home,
                away=away,
                commence_time=commence,
                lines=lines,
            ))
        return events
