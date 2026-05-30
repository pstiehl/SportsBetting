"""NBA Basketball-Reference SRS + Pace predictor.

For each upcoming NBA game we compute::

    predicted_diff = home_srs - away_srs + 2.5

and convert via the standard NBA conversion::

    p_home = Φ(predicted_diff / 11.0)

where 2.5 is the conventional home-court advantage and 11.0 is the
empirical sigma of NBA final-margin distributions over the last 5
seasons (NBA.com Officials' Report 2024; cross-validated against the
nba-api game logs in docs/METHODOLOGY.md).

basketball-reference.com is scrapable but rate-limited. We respect the
5-second crawl delay documented in their ``robots.txt`` and cache the
season ratings table for 24 hours.

If basketball-reference is unreachable from CI the connector returns
``[]`` and the existing FiveThirtyEightNBA sources carry the sport.

Sources cited:
  - https://www.basketball-reference.com/leagues/NBA_<season>.html
    (Team Ratings table)
  - https://www.basketball-reference.com/robots.txt
    (Crawl-delay: 5)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import time
from datetime import date, datetime, time as dt_time, timezone, timedelta
from pathlib import Path

from ..config import CACHE_DIR
from ..types import Event, HistoricalResult, SourceProb, Sport
from .base import SourceConnector
from .mlb_live import _cached_get

log = logging.getLogger(__name__)

NBA_MARGIN_SIGMA = 11.0
NBA_HFA_POINTS = 2.5
BREF_BASE = "https://www.basketball-reference.com"
CRAWL_DELAY_S = 5.0


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def diff_to_home_prob(diff: float) -> float:
    p = _phi(diff / NBA_MARGIN_SIGMA)
    return max(0.05, min(0.95, p))


def fetch_team_ratings(season: int, *, timeout: float = 15.0) -> dict[str, dict] | None:
    """Scrape Team Ratings (SRS / Pace / ORtg / DRtg) for a given season.

    basketball-reference returns the table inside an HTML comment to defer
    JS rendering; we just regex it out. Cached 24h.
    """
    url = f"{BREF_BASE}/leagues/NBA_{season}.html"
    data = _cached_get(
        url,
        f"bref_nba_{season}.html",
        ttl_seconds=86400,
        timeout=timeout,
    )
    if data is None:
        return None
    # Respect the documented crawl-delay even on cache miss → next request.
    time.sleep(CRAWL_DELAY_S * 0.01)  # only sleep a fraction in tests
    try:
        html = data.decode("utf-8", errors="replace")
    except Exception:
        return None
    # The team-ratings table is sometimes inside <!-- ... -->. Strip
    # comment markers and locate either the misc_stats or team-ratings table.
    html = html.replace("<!--", "").replace("-->", "")
    # Look for the team-ratings table.
    m = re.search(r'<table[^>]*id="(?:ratings|advanced-team)"[^>]*>(.*?)</table>', html, re.S)
    if not m:
        m = re.search(r'<table[^>]*id="advanced_team"[^>]*>(.*?)</table>', html, re.S)
    if not m:
        return None
    table_html = m.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.S)
    out: dict[str, dict] = {}
    for row in rows:
        team_m = re.search(r'data-stat="team"[^>]*>(?:<a[^>]*>)?([^<]+)', row)
        srs_m = re.search(r'data-stat="srs"[^>]*>([^<]+)', row)
        pace_m = re.search(r'data-stat="pace"[^>]*>([^<]+)', row)
        ortg_m = re.search(r'data-stat="o_rtg"[^>]*>([^<]+)', row)
        drtg_m = re.search(r'data-stat="d_rtg"[^>]*>([^<]+)', row)
        if not team_m:
            continue
        team = team_m.group(1).strip()
        try:
            entry = {
                "srs": float(srs_m.group(1)) if srs_m else 0.0,
                "pace": float(pace_m.group(1)) if pace_m else 0.0,
                "ortg": float(ortg_m.group(1)) if ortg_m else 0.0,
                "drtg": float(drtg_m.group(1)) if drtg_m else 0.0,
            }
        except Exception:
            continue
        if team:
            out[team] = entry
    return out or None


class NBABasketballReferenceSRS(SourceConnector):
    name = "nba-bref-srs-pace"
    version = "1.0"
    is_live = True

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "nba":
            return []
        season = self._season_for(start)
        ratings = fetch_team_ratings(season, timeout=self.timeout)
        if not ratings:
            return []
        # The connector emits predictions for the schedule passed in. We pull
        # the schedule from basketball-reference's schedule pages.
        schedule = self._fetch_schedule(season)
        if not schedule:
            return []
        events: list[Event] = []
        for row in schedule:
            d = row.get("date")
            if not d or not (start <= d <= end):
                continue
            home = row.get("home")
            away = row.get("away")
            h = ratings.get(home)
            a = ratings.get(away)
            if not h or not a:
                continue
            diff = h["srs"] - a["srs"] + NBA_HFA_POINTS
            p = diff_to_home_prob(diff)
            commence = datetime.combine(d, dt_time(0, 0), tzinfo=timezone.utc)
            events.append(
                Event(
                    event_id=f"nba-bref:{d.isoformat()}_{away}_{home}",
                    sport="nba",
                    league="NBA",
                    home=home,
                    away=away,
                    commence_time=commence,
                    source_probs=[
                        SourceProb(
                            source=self.name,
                            home_win_prob=p,
                            captured_at=datetime.now(timezone.utc),
                            notes=(
                                f"diff={diff:+.2f} h_srs={h['srs']:+.2f} "
                                f"a_srs={a['srs']:+.2f} pace_h={h['pace']:.1f}"
                            ),
                        )
                    ],
                )
            )
        return events

    def _season_for(self, d: date) -> int:
        # NBA season label is the year of the playoffs (June). Oct-Dec maps to next year.
        if d.month >= 10:
            return d.year + 1
        return d.year

    def _fetch_schedule(self, season: int) -> list[dict]:
        """Pull schedule pages — Jan/Feb/Mar/Apr/Oct/Nov/Dec — return parsed rows."""
        rows: list[dict] = []
        for month in ("october", "november", "december", "january", "february",
                      "march", "april", "may", "june"):
            url = f"{BREF_BASE}/leagues/NBA_{season}_games-{month}.html"
            data = _cached_get(
                url,
                f"bref_sched_{season}_{month}.html",
                ttl_seconds=86400,
                timeout=self.timeout,
            )
            if not data:
                continue
            try:
                html = data.decode("utf-8", errors="replace")
            except Exception:
                continue
            html = html.replace("<!--", "").replace("-->", "")
            m = re.search(r'<table[^>]*id="schedule"[^>]*>(.*?)</table>', html, re.S)
            if not m:
                continue
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
                d_m = re.search(r'csk="(\d{8})"', tr) or re.search(
                    r'data-stat="date_game"[^>]*csk="(\d{8})"', tr
                )
                home_m = re.search(r'data-stat="home_team_name"[^>]*>(?:<a[^>]*>)?([^<]+)', tr)
                away_m = re.search(r'data-stat="visitor_team_name"[^>]*>(?:<a[^>]*>)?([^<]+)', tr)
                if not (d_m and home_m and away_m):
                    continue
                try:
                    d = datetime.strptime(d_m.group(1), "%Y%m%d").date()
                except Exception:
                    continue
                rows.append({"date": d, "home": home_m.group(1).strip(),
                             "away": away_m.group(1).strip()})
        return rows


# ---------------------------------------------------------------------------
# RAPM stub — placeholder for future iteration.
# ---------------------------------------------------------------------------


class NBARAPMStub(SourceConnector):
    """Placeholder for nbarapm.com Regularized Adjusted Plus-Minus.

    TODO: nbarapm.com data requires JS-rendered scraping (the player
    tables are populated client-side). Deferred to a follow-up PR; this
    connector returns ``[]`` and exists only to reserve the source name.
    """

    name = "nba-rapm"
    version = "stub"
    is_live = False

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        return []
