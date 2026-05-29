"""NBA historical connector — FiveThirtyEight's classic Elo dataset.

URL: https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv

Coverage: 1946-47 through 2014-15 (538 retired their live MLB/NBA Elo
endpoints in 2023; the static CSV remains in the data repo).

Why this matters for the backtest:
  - Each game has a ``forecast`` column = pre-game Elo win probability
    for the team in that row. This is a legitimate, no-outcome-leakage
    source probability we can backtest against.
  - Each game row is duplicated (one per team) and contains the game
    result, so we get realised outcomes for free.
  - No moneyline odds in this dataset — Kelly/ROI scoring will be
    *skipped* (no_market_price) on these events. We still get Brier
    score on the 538 forecast, which validates the calibration pipeline.

For modern NBA odds + results we'd need either (a) Phil's paid Odds API
historical archive, or (b) a sportsbookreviewsonline scrape. Both are
deferred to a follow-up PR — see PHIL_PLAN.md.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, time, timezone

import httpx

from ..config import CACHE_DIR
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

CSV_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/"
    "master/nba-elo/nbaallelo.csv"
)
CACHE_FILE = "fivethirtyeight_nba_elo.csv"


class FiveThirtyEightNBAHistorical(SourceConnector):
    name = "fivethirtyeight-nba-elo"
    version = "static-2015"
    is_live = True

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport is not None and sport != "nba":
            return []
        events, _ = self._load_range(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._load_range(start, end)
        return results

    # ----- internals ---------------------------------------------------

    def _load_csv(self) -> list[dict]:
        cache_path = CACHE_DIR / CACHE_FILE
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    r = client.get(CSV_URL)
                    r.raise_for_status()
                cache_path.write_bytes(r.content)
            except Exception as e:  # noqa: BLE001
                log.warning("538 NBA Elo download failed: %s", e)
                return []
        with open(cache_path) as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _load_range(
        self, start: date, end: date
    ) -> tuple[list[Event], list[HistoricalResult]]:
        rows = self._load_csv()
        events: dict[str, Event] = {}
        results: dict[str, HistoricalResult] = {}
        for row in rows:
            # Each game appears twice — once per team. We only keep the
            # "_iscopy=0" row to avoid double-counting.
            if (row.get("_iscopy") or "0") != "0":
                continue
            try:
                game_date = datetime.strptime(row["date_game"], "%m/%d/%Y").date()
            except Exception:
                continue
            if not (start <= game_date <= end):
                continue
            try:
                forecast = float(row.get("forecast") or "")
            except Exception:
                forecast = None
            home_loc = (row.get("game_location") or "").upper()
            team = row.get("fran_id") or row.get("team_id")
            opp = row.get("opp_fran") or row.get("opp_id")
            game_result = (row.get("game_result") or "").upper()
            if not team or not opp:
                continue

            # 538 row is from the perspective of `team`. If game_location==H,
            # then `team` is home. Otherwise, away.
            if home_loc == "H":
                home, away = team, opp
                home_won = (game_result == "W")
                home_prob = forecast
            else:
                home, away = opp, team
                home_won = (game_result == "L")
                home_prob = (1.0 - forecast) if forecast is not None else None

            game_id = row.get("game_id") or f"{game_date}_{home}_{away}"
            event_id = f"538nba:{game_id}"
            if event_id in events:
                continue

            commence = datetime.combine(game_date, time(20, 0), tzinfo=timezone.utc)
            source_probs: list[SourceProb] = []
            if home_prob is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nba-elo",
                        home_win_prob=max(0.001, min(0.999, home_prob)),
                        captured_at=commence,
                        notes="538 Elo pre-game forecast",
                    )
                )
            events[event_id] = Event(
                event_id=event_id,
                sport="nba",
                league="NBA",
                home=home,
                away=away,
                commence_time=commence,
                source_probs=source_probs,
            )
            results[event_id] = HistoricalResult(
                event_id=event_id,
                sport="nba",
                home=home,
                away=away,
                commence_time=commence,
                home_won=home_won,
            )
        return list(events.values()), list(results.values())
