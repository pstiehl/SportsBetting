"""Polymarket connector — pulls publicly traded probabilities on sports events.

Docs: https://docs.polymarket.com/
The Gamma markets endpoint returns active markets with last-trade prices,
which we treat as crowd-implied probabilities.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone

import httpx

from ..config import CACHE_DIR
from ..types import Event, Sport, SourceProb
from .base import SourceConnector

log = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"

# Polymarket tag/topic to our sport mapping (best-effort string match).
SPORT_HINTS: dict[Sport, list[str]] = {
    "nfl": ["nfl", "national football league"],
    "nba": ["nba"],
    "mlb": ["mlb", "baseball"],
    "nhl": ["nhl", "hockey"],
    "cfb": ["college football", "ncaaf"],
    "cbb": ["college basketball", "ncaab", "march madness"],
}


class Polymarket(SourceConnector):
    name = "polymarket"
    version = "gamma-v1"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        params = {
            "active": "true",
            "closed": "false",
            "limit": 200,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.get(GAMMA_URL, params=params)
                r.raise_for_status()
                markets = r.json()
        except Exception as e:  # noqa: BLE001
            log.debug("polymarket fetch failed: %s", e)
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_DIR / "polymarket.json", "w") as f:
            json.dump(markets, f, indent=2)
        sports = [sport] if sport else list(SPORT_HINTS.keys())
        return self._parse(markets, sports)

    @staticmethod
    def _classify_sport(question: str, tags: list[str], sports: list[Sport]) -> Sport | None:
        text = (question + " " + " ".join(tags)).lower()
        for s in sports:
            for hint in SPORT_HINTS[s]:
                if hint in text:
                    return s
        return None

    @staticmethod
    def _parse_h2h(question: str) -> tuple[str, str] | None:
        """Extract 'TeamA vs TeamB' style pairs from a Polymarket question."""
        m = re.search(r"^([^?]+?)\s+(?:vs\.?|@)\s+([^?]+?)(?:[\?\.\:].*)?$", question, re.I)
        if not m:
            return None
        return m.group(1).strip(), m.group(2).strip()

    def _parse(self, markets: list[dict], sports: list[Sport]) -> list[Event]:
        now = datetime.now(timezone.utc)
        out: list[Event] = []
        for m in markets:
            question = (m.get("question") or "").strip()
            if not question:
                continue
            tags = [t.get("label", "") for t in m.get("tags", []) if isinstance(t, dict)]
            sport = self._classify_sport(question, tags, sports)
            if sport is None:
                continue
            pair = self._parse_h2h(question)
            if not pair:
                continue
            home, away = pair  # treat first-named team as "home" by convention
            try:
                last_price = float(m.get("lastTradePrice") or m.get("outcomePrices", [0.5])[0])
            except Exception:
                last_price = 0.5
            last_price = max(0.001, min(0.999, last_price))
            try:
                end_dt = m.get("endDate") or m.get("end_date") or ""
                commence = (
                    datetime.fromisoformat(end_dt.replace("Z", "+00:00"))
                    if end_dt
                    else now
                )
            except Exception:
                commence = now
            event_id = f"polymarket:{m.get('id') or m.get('conditionId') or question[:32]}"
            out.append(
                Event(
                    event_id=event_id,
                    sport=sport,
                    league=sport.upper(),
                    home=home,
                    away=away,
                    commence_time=commence,
                    source_probs=[
                        SourceProb(
                            source="polymarket",
                            home_win_prob=last_price,
                            captured_at=now,
                            notes="last trade price",
                        )
                    ],
                )
            )
        return out
