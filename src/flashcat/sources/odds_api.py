"""The Odds API connector — moneylines across many books.

Docs: https://the-odds-api.com/liveapi/guides/v4/
Free tier requires an API key in env var THE_ODDS_API_KEY.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from ..config import CACHE_DIR, SAMPLES_DIR, the_odds_api_key
from ..types import BookLine, Event, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "nfl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "cfb": "americanfootball_ncaaf",
    "cbb": "basketball_ncaab",
}


class TheOddsAPI(SourceConnector):
    name = "the-odds-api"
    version = "v4"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.api_key = the_odds_api_key()

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if not self.api_key:
            log.info("THE_ODDS_API_KEY not set; falling back to committed sample data")
            return self._load_sample()
        sports = [sport] if sport else list(SPORT_KEYS.keys())
        events: list[Event] = []
        for s in sports:
            try:
                events.extend(self._fetch_sport(s))
            except Exception as e:  # noqa: BLE001
                log.warning("the-odds-api fetch failed for %s: %s", s, e)
        return events

    def _fetch_sport(self, sport: Sport) -> list[Event]:
        sport_key = SPORT_KEYS[sport]
        url = f"{BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        # Cache for diagnostics
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"odds_api_{sport}.json", "w") as f:
            json.dump(data, f, indent=2)
        return self._parse_payload(data, sport)

    @staticmethod
    def _parse_payload(data: list[dict], sport: Sport) -> list[Event]:
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        for game in data:
            event_id = f"oddsapi:{game['id']}"
            try:
                commence = datetime.fromisoformat(
                    game["commence_time"].replace("Z", "+00:00")
                )
            except Exception:
                commence = now
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            lines: list[BookLine] = []
            for bk in game.get("bookmakers", []):
                book = bk.get("key", "unknown")
                for market in bk.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", 0)
                        side = "home" if name == home else "away"
                        lines.append(
                            BookLine(
                                book=book,
                                side=side,
                                american=int(price),
                                captured_at=now,
                                is_opening=False,
                            )
                        )
            out.append(
                Event(
                    event_id=event_id,
                    sport=sport,
                    league=sport.upper(),
                    home=home,
                    away=away,
                    commence_time=commence,
                    lines=lines,
                )
            )
        return out

    @staticmethod
    def _load_sample() -> list[Event]:
        sample_path = SAMPLES_DIR / "odds_api_sample.json"
        if not sample_path.exists():
            log.debug("no odds_api sample found at %s", sample_path)
            return []
        with open(sample_path) as f:
            data = json.load(f)
        out: list[Event] = []
        for sport, payload in data.items():
            out.extend(TheOddsAPI._parse_payload(payload, sport))  # type: ignore[arg-type]
        return out
