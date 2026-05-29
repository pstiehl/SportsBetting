"""The Odds API connector — moneylines across many books.

Docs: https://the-odds-api.com/liveapi/guides/v4/
Free tier requires an API key in env var THE_ODDS_API_KEY.

Behavior change (post-fake-data incident):
- If no API key is set, this connector returns ``[]`` unless
  ``FLASHCAT_USE_SAMPLES=1`` is set (opt-in for offline local dev).
- Active-sport detection: ``active_sports()`` queries ``/v4/sports?all=false``
  so the build pipeline only requests sports that are in season today.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import httpx

from ..config import CACHE_DIR, SAMPLES_DIR, the_odds_api_key, use_samples_fallback
from ..types import BookLine, Event, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# Canonical sport-key map. Tennis keys are seasonal (e.g. french_open),
# so for tennis we resolve at runtime via ``active_sports()``.
SPORT_KEYS = {
    "nfl": "americanfootball_nfl",
    "nba": "basketball_nba",
    "mlb": "baseball_mlb",
    "nhl": "icehockey_nhl",
    "cfb": "americanfootball_ncaaf",
    "cbb": "basketball_ncaab",
}

# Group-prefix → our sport tag for runtime tennis discovery.
TENNIS_PREFIXES = ("tennis_atp", "tennis_wta")


def _tennis_tag(sport_key: str) -> Sport | None:
    if sport_key.startswith("tennis_atp"):
        return "atp"
    if sport_key.startswith("tennis_wta"):
        return "wta"
    return None


class TheOddsAPI(SourceConnector):
    name = "the-odds-api"
    version = "v4"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.api_key = the_odds_api_key()

    # ----- discovery ---------------------------------------------------

    def active_sports(self) -> list[tuple[Sport, str]]:
        """Return list of ``(our_sport_tag, oddsapi_key)`` currently in season.

        Falls back to the static SPORT_KEYS dict if no API key is available.
        Tennis keys are dynamic — we only emit them when the API reports them
        active.
        """
        static = [(s, SPORT_KEYS[s]) for s in SPORT_KEYS]
        if not self.api_key:
            log.info(
                "THE_ODDS_API_KEY not set; can't discover active sports — using static list"
            )
            return static
        url = f"{BASE_URL}/sports/"
        params = {"apiKey": self.api_key, "all": "false"}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("Odds API /sports list failed: %s — falling back to static", e)
            return static
        out: list[tuple[Sport, str]] = []
        reverse = {v: k for k, v in SPORT_KEYS.items()}
        for s in data:
            key = s.get("key", "")
            if not s.get("active"):
                continue
            if key in reverse:
                out.append((reverse[key], key))
                continue
            t = _tennis_tag(key)
            if t is not None:
                out.append((t, key))
        # Dedupe — many tennis tournaments map to same atp/wta tag; we'll
        # union them at fetch time by storing the underlying keys.
        return out

    # ----- fetch -------------------------------------------------------

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if not self.api_key:
            if use_samples_fallback():
                log.warning(
                    "THE_ODDS_API_KEY not set; FLASHCAT_USE_SAMPLES=1 → loading committed samples"
                )
                return self._load_sample()
            log.warning(
                "THE_ODDS_API_KEY not set; OddsAPI returning [] "
                "(set FLASHCAT_USE_SAMPLES=1 for offline dev)"
            )
            return []

        # Pull active sport list once and filter.
        active = self.active_sports()
        if sport:
            active = [(t, k) for (t, k) in active if t == sport]
        events: list[Event] = []
        for tag, key in active:
            try:
                events.extend(self._fetch_sport_key(tag, key))
            except Exception as e:  # noqa: BLE001
                log.warning("the-odds-api fetch failed for %s/%s: %s", tag, key, e)
        return events

    def _fetch_sport_key(self, sport: Sport, sport_key: str) -> list[Event]:
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
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / f"odds_api_{sport_key}.json", "w") as f:
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
            # Tennis events don't have home/away — Odds API still returns one
            # side as "home_team". We keep the convention but the meaning is
            # just "team listed first".
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
        # Prefer the explicit `.example.json` quarantined file when present.
        example_path = SAMPLES_DIR / "odds_api_sample.example.json"
        path = example_path if example_path.exists() else sample_path
        if not path.exists():
            log.debug("no odds_api sample found at %s or %s", sample_path, example_path)
            return []
        with open(path) as f:
            data = json.load(f)
        out: list[Event] = []
        for sport, payload in data.items():
            out.extend(TheOddsAPI._parse_payload(payload, sport))  # type: ignore[arg-type]
        return out
