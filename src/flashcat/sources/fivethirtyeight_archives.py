"""FiveThirtyEight historical Elo/Forecast archives.

538 retired their live MLB/NBA/NFL prediction pages in 2023, but the final
CSV snapshots were captured by the Wayback Machine. We pull from there and
cache locally under ``data/cache/``.

Each connector:
  - returns Events with ``source_probs`` containing the model's pre-game
    win probability (one ``SourceProb`` per model variant where available),
  - implements ``load_results`` so the backtest grader has outcomes,
  - emits a stable ``event_id`` of the form ``538<sport>:<season>_<date>_<team1>_<team2>``.

Sources used:
  - MLB:  Elo (``elo_prob1``) + pitcher-adjusted rating (``rating_prob1``).
            Coverage 1871-2023.
  - NFL:  Elo (``elo_prob1``) + QB-adjusted (``qbelo_prob1``).
            Coverage 1920-2023.
  - NBA:  Elo (``elo_prob1``) + CARM-Elo (``carm-elo_prob1``) + RAPTOR
            (``raptor_prob1``). Coverage 1946-2022.

We never recompute outcomes — outcomes come from the CSV ``score1``/``score2``
columns, which 538 filled in *after* the game. The pre-game ``elo_prob1`` was
published *before* the game. Walk-forward holds because the CSV row is the
historical artifact, not a re-fit.
"""

from __future__ import annotations

import csv
import logging
from datetime import date, datetime, time, timezone

import httpx

from ..config import CACHE_DIR
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

# Wayback-mirrored 538 archive URLs.
MLB_URL = "https://web.archive.org/web/2023/https://projects.fivethirtyeight.com/mlb-api/mlb_elo.csv"
NFL_URL = "https://web.archive.org/web/2023/https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv"
NBA_URL = "https://web.archive.org/web/2023/https://projects.fivethirtyeight.com/nba-model/nba_elo.csv"


def _safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _clamp(p: float) -> float:
    return max(0.001, min(0.999, p))


def _download_csv(url: str, cache_file: str, timeout: float = 90.0) -> list[dict]:
    cache_path = CACHE_DIR / cache_file
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()
            cache_path.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            log.warning("538 archive download failed for %s: %s", url, e)
            return []
    try:
        with open(cache_path, newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:  # noqa: BLE001
        log.warning("538 archive CSV parse failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# MLB
# ---------------------------------------------------------------------------


class FiveThirtyEightMLBElo(SourceConnector):
    """538 MLB Elo + pitcher-adjusted rating archive."""

    name = "fivethirtyeight-mlb-elo"
    version = "wayback-2023"
    is_live = True

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        events, _ = self._load_range(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._load_range(start, end)
        return results

    def _load_range(
        self, start: date, end: date
    ) -> tuple[list[Event], list[HistoricalResult]]:
        rows = _download_csv(MLB_URL, "538_mlb_elo.csv", timeout=self.timeout)
        events: list[Event] = []
        results: list[HistoricalResult] = []
        for row in rows:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            team1 = row.get("team1")
            team2 = row.get("team2")
            if not team1 or not team2:
                continue
            # team1 is home in the 538 convention (per the dataset readme).
            home, away = team1, team2
            elo_p1 = _safe_float(row.get("elo_prob1"))
            rating_p1 = _safe_float(row.get("rating_prob1"))
            score1 = _safe_float(row.get("score1"))
            score2 = _safe_float(row.get("score2"))

            commence = datetime.combine(d, time(19, 0), tzinfo=timezone.utc)
            event_id = f"538mlb:{row.get('season','?')}_{d}_{home}_{away}"
            source_probs: list[SourceProb] = []
            if elo_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-mlb-elo",
                        home_win_prob=_clamp(elo_p1),
                        captured_at=commence,
                        notes="538 pre-game team Elo (no pitcher adj)",
                    )
                )
            if rating_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-mlb-rating",
                        home_win_prob=_clamp(rating_p1),
                        captured_at=commence,
                        notes="538 pre-game pitcher-adjusted rating",
                    )
                )
            if source_probs:
                events.append(
                    Event(
                        event_id=event_id,
                        sport="mlb",
                        league="MLB",
                        home=home,
                        away=away,
                        commence_time=commence,
                        source_probs=source_probs,
                    )
                )
            if score1 is not None and score2 is not None:
                results.append(
                    HistoricalResult(
                        event_id=event_id,
                        sport="mlb",
                        home=home,
                        away=away,
                        commence_time=commence,
                        home_won=score1 > score2,
                        home_score=int(score1),
                        away_score=int(score2),
                    )
                )
        return events, results


# ---------------------------------------------------------------------------
# NFL
# ---------------------------------------------------------------------------


class FiveThirtyEightNFLElo(SourceConnector):
    """538 NFL Elo + QB-adjusted Elo archive."""

    name = "fivethirtyeight-nfl-elo"
    version = "wayback-2023"
    is_live = True

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "nfl":
            return []
        events, _ = self._load_range(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._load_range(start, end)
        return results

    def _load_range(
        self, start: date, end: date
    ) -> tuple[list[Event], list[HistoricalResult]]:
        rows = _download_csv(NFL_URL, "538_nfl_elo.csv", timeout=self.timeout)
        events: list[Event] = []
        results: list[HistoricalResult] = []
        for row in rows:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            home = row.get("team1")
            away = row.get("team2")
            if not home or not away:
                continue
            elo_p1 = _safe_float(row.get("elo_prob1"))
            qbelo_p1 = _safe_float(row.get("qbelo_prob1"))
            score1 = _safe_float(row.get("score1"))
            score2 = _safe_float(row.get("score2"))
            commence = datetime.combine(d, time(20, 0), tzinfo=timezone.utc)
            event_id = f"538nfl:{row.get('season','?')}_{d}_{home}_{away}"
            source_probs: list[SourceProb] = []
            if elo_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nfl-elo",
                        home_win_prob=_clamp(elo_p1),
                        captured_at=commence,
                        notes="538 pre-game team Elo",
                    )
                )
            if qbelo_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nfl-qbelo",
                        home_win_prob=_clamp(qbelo_p1),
                        captured_at=commence,
                        notes="538 QB-adjusted Elo",
                    )
                )
            if source_probs:
                events.append(
                    Event(
                        event_id=event_id,
                        sport="nfl",
                        league="NFL",
                        home=home,
                        away=away,
                        commence_time=commence,
                        source_probs=source_probs,
                    )
                )
            if score1 is not None and score2 is not None:
                results.append(
                    HistoricalResult(
                        event_id=event_id,
                        sport="nfl",
                        home=home,
                        away=away,
                        commence_time=commence,
                        home_won=score1 > score2,
                        home_score=int(score1),
                        away_score=int(score2),
                    )
                )
        return events, results


# ---------------------------------------------------------------------------
# NBA (modern 538 Elo + CARM + RAPTOR — supersedes the static nbaallelo.csv)
# ---------------------------------------------------------------------------


class FiveThirtyEightNBAModern(SourceConnector):
    """538 NBA Elo + CARM-Elo + RAPTOR archive (2014-15 onward)."""

    name = "fivethirtyeight-nba-modern"
    version = "wayback-2023"
    is_live = True

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "nba":
            return []
        events, _ = self._load_range(start, end)
        return events

    def load_results(self, start: date, end: date) -> list[HistoricalResult]:
        _, results = self._load_range(start, end)
        return results

    def _load_range(
        self, start: date, end: date
    ) -> tuple[list[Event], list[HistoricalResult]]:
        rows = _download_csv(NBA_URL, "538_nba_elo.csv", timeout=self.timeout)
        events: list[Event] = []
        results: list[HistoricalResult] = []
        for row in rows:
            try:
                d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            if not (start <= d <= end):
                continue
            home = row.get("team1")
            away = row.get("team2")
            if not home or not away:
                continue
            elo_p1 = _safe_float(row.get("elo_prob1"))
            carm_p1 = _safe_float(row.get("carm-elo_prob1"))
            raptor_p1 = _safe_float(row.get("raptor_prob1"))
            score1 = _safe_float(row.get("score1"))
            score2 = _safe_float(row.get("score2"))
            commence = datetime.combine(d, time(20, 0), tzinfo=timezone.utc)
            event_id = f"538nba:{row.get('season','?')}_{d}_{home}_{away}"
            source_probs: list[SourceProb] = []
            if elo_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nba-elo-modern",
                        home_win_prob=_clamp(elo_p1),
                        captured_at=commence,
                        notes="538 pre-game team Elo",
                    )
                )
            if carm_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nba-carm",
                        home_win_prob=_clamp(carm_p1),
                        captured_at=commence,
                        notes="538 CARM-Elo",
                    )
                )
            if raptor_p1 is not None:
                source_probs.append(
                    SourceProb(
                        source="fivethirtyeight-nba-raptor",
                        home_win_prob=_clamp(raptor_p1),
                        captured_at=commence,
                        notes="538 RAPTOR",
                    )
                )
            if source_probs:
                events.append(
                    Event(
                        event_id=event_id,
                        sport="nba",
                        league="NBA",
                        home=home,
                        away=away,
                        commence_time=commence,
                        source_probs=source_probs,
                    )
                )
            if score1 is not None and score2 is not None:
                results.append(
                    HistoricalResult(
                        event_id=event_id,
                        sport="nba",
                        home=home,
                        away=away,
                        commence_time=commence,
                        home_won=score1 > score2,
                        home_score=int(score1),
                        away_score=int(score2),
                    )
                )
        return events, results
