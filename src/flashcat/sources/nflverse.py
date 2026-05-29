"""NFLverse / nfl_data_py connector — historical NFL results + odds.

Used to build the backtest dataset. If nfl_data_py is not installed, this
falls back to loading data/samples/nfl_<season>.json (committed sample).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from ..config import SAMPLES_DIR
from ..types import BookLine, Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)


class NFLverseHistorical(SourceConnector):
    """Historical NFL schedules + spreads + moneylines via nfl_data_py."""

    name = "nflverse"
    version = "nfl_data_py"
    is_live = True

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        if sport and sport != "nfl":
            return []
        seasons = sorted({start.year, end.year})
        df = self._load_schedules(seasons)
        if df is None:
            return self._load_sample(seasons)
        events: list[Event] = []
        for row in df.to_dict(orient="records"):
            game_date = row.get("gameday")
            if not game_date:
                continue
            try:
                d = (
                    datetime.fromisoformat(str(game_date)).date()
                    if "-" in str(game_date)
                    else datetime.strptime(str(game_date), "%Y%m%d").date()
                )
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            ev = self._row_to_event(row, d)
            if ev:
                events.append(ev)
        return events

    @staticmethod
    def _load_schedules(seasons: list[int]) -> Any:
        try:
            import nfl_data_py as nfl  # type: ignore
        except Exception:
            log.info("nfl_data_py not available; falling back to samples")
            return None
        try:
            df = nfl.import_schedules(seasons)
            return df
        except Exception as e:  # noqa: BLE001
            log.warning("nfl_data_py import_schedules failed: %s", e)
            return None

    @staticmethod
    def _row_to_event(row: dict, d: date) -> Event | None:
        home = row.get("home_team")
        away = row.get("away_team")
        if not home or not away:
            return None
        game_id = row.get("game_id") or f"{d}_{home}_{away}"
        commence = datetime.combine(d, time(20, 0), tzinfo=timezone.utc)
        lines: list[BookLine] = []
        # nflverse schedule has 'home_moneyline' and 'away_moneyline' (closing-ish).
        hm = row.get("home_moneyline")
        am = row.get("away_moneyline")
        opening_hm = row.get("home_moneyline_open") or row.get("opening_home_moneyline")
        opening_am = row.get("away_moneyline_open") or row.get("opening_away_moneyline")
        if hm and am:
            try:
                lines.append(
                    BookLine(
                        book="consensus-close",
                        side="home",
                        american=int(hm),
                        captured_at=commence,
                        is_opening=False,
                    )
                )
                lines.append(
                    BookLine(
                        book="consensus-close",
                        side="away",
                        american=int(am),
                        captured_at=commence,
                        is_opening=False,
                    )
                )
            except Exception:
                pass
        if opening_hm and opening_am:
            try:
                lines.append(
                    BookLine(
                        book="consensus-open",
                        side="home",
                        american=int(opening_hm),
                        captured_at=commence,
                        is_opening=True,
                    )
                )
                lines.append(
                    BookLine(
                        book="consensus-open",
                        side="away",
                        american=int(opening_am),
                        captured_at=commence,
                        is_opening=True,
                    )
                )
            except Exception:
                pass
        # Treat closing market as a source prob too — devig later in pipeline.
        return Event(
            event_id=f"nflverse:{game_id}",
            sport="nfl",
            league="NFL",
            home=str(home),
            away=str(away),
            commence_time=commence,
            lines=lines,
        )

    @staticmethod
    def _load_sample(seasons: list[int]) -> list[Event]:
        out: list[Event] = []
        for season in seasons:
            p = SAMPLES_DIR / f"nfl_{season}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            for row in data:
                d_str = row.get("gameday", "")
                try:
                    d = datetime.fromisoformat(d_str).date()
                except Exception:
                    continue
                ev = NFLverseHistorical._row_to_event(row, d)
                if ev:
                    out.append(ev)
        return out

    @staticmethod
    def load_results(start: date, end: date) -> list[HistoricalResult]:
        """Realized results — used by the backtest grader."""
        try:
            import nfl_data_py as nfl  # type: ignore
        except Exception:
            return NFLverseHistorical._load_sample_results(start, end)
        try:
            df = nfl.import_schedules(sorted({start.year, end.year}))
        except Exception:
            return NFLverseHistorical._load_sample_results(start, end)
        out: list[HistoricalResult] = []
        for row in df.to_dict(orient="records"):
            g = row.get("gameday")
            home_s = row.get("home_score")
            away_s = row.get("away_score")
            if home_s is None or away_s is None or g is None:
                continue
            try:
                d = (
                    datetime.fromisoformat(str(g)).date()
                    if "-" in str(g)
                    else datetime.strptime(str(g), "%Y%m%d").date()
                )
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            home = row.get("home_team")
            away = row.get("away_team")
            game_id = row.get("game_id") or f"{d}_{home}_{away}"
            out.append(
                HistoricalResult(
                    event_id=f"nflverse:{game_id}",
                    sport="nfl",
                    home=str(home),
                    away=str(away),
                    commence_time=datetime.combine(d, time(20, 0), tzinfo=timezone.utc),
                    home_won=int(home_s) > int(away_s),
                    home_score=int(home_s),
                    away_score=int(away_s),
                )
            )
        return out

    @staticmethod
    def _load_sample_results(start: date, end: date) -> list[HistoricalResult]:
        out: list[HistoricalResult] = []
        for season in sorted({start.year, end.year}):
            p = SAMPLES_DIR / f"nfl_{season}.json"
            if not p.exists():
                continue
            with open(p) as f:
                data = json.load(f)
            for row in data:
                if row.get("home_score") is None or row.get("away_score") is None:
                    continue
                try:
                    d = datetime.fromisoformat(row["gameday"]).date()
                except Exception:
                    continue
                if not (start <= d <= end):
                    continue
                game_id = row.get("game_id") or f"{d}_{row.get('home_team')}_{row.get('away_team')}"
                out.append(
                    HistoricalResult(
                        event_id=f"nflverse:{game_id}",
                        sport="nfl",
                        home=row.get("home_team", ""),
                        away=row.get("away_team", ""),
                        commence_time=datetime.combine(
                            d, time(20, 0), tzinfo=timezone.utc
                        ),
                        home_won=int(row["home_score"]) > int(row["away_score"]),
                        home_score=int(row["home_score"]),
                        away_score=int(row["away_score"]),
                    )
                )
        return out
