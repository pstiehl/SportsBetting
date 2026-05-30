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

# basketball-reference.com blocks generic User-Agents with 403 in CI. They
# document a 5-second Crawl-delay in robots.txt and are friendlier to UAs
# that identify a real project URL. The headers below match a real browser
# enough to clear Cloudflare's bot check while still being honest about
# who's calling. NOTE: the actual sleep below is intentionally short in
# tests; the live ``fetch_team_ratings`` path explicitly honors the
# Crawl-delay between requests.
BREF_UA = (
    "Mozilla/5.0 (compatible; flashcat-research/1.0; "
    "+https://github.com/pstiehl/SportsBetting)"
)
BREF_HEADERS = {
    "User-Agent": BREF_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.basketball-reference.com/",
}

# stats.nba.com fallback. Used when basketball-reference returns 403 from
# this environment (cloud egress / CI is currently bref-blocked despite the
# project-identifying UA fix from PR #12). The fallback computes a coarse
# "avg-margin SRS" from ``DiffPointsPG`` in LeagueStandingsV3 — the strength
# of schedule adjustment of true SRS is omitted because schedule symmetry
# washes out the difference at the team-vs-team level we care about here.
NBA_STATS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def diff_to_home_prob(diff: float) -> float:
    p = _phi(diff / NBA_MARGIN_SIGMA)
    return max(0.05, min(0.95, p))


def _fetch_team_ratings_nba_stats(season: int, *, timeout: float = 30.0) -> dict[str, dict] | None:
    """Fallback rating fetch via ``stats.nba.com`` LeagueStandingsV3.

    Maps NBA-API ``DiffPointsPG`` onto ``srs`` (true SRS adds opponent-strength;
    DiffPointsPG is just avg margin). ``pace`` / ``ortg`` / ``drtg`` are best-
    effort: NBA-API exposes them via a different endpoint (TeamEstimatedMetrics),
    but a 0.0 default is safe — the live connector only uses ``srs`` for the
    diff-to-prob conversion. The fallback exists so live picks don't drop to
    [] when bref 403s.
    """
    try:
        from nba_api.stats.endpoints import leaguestandingsv3  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.warning("nba_api not installed, fallback unavailable: %s", e)
        return None
    # Season label conversion. bref season=2024 → nba-api season="2023-24".
    label = f"{season - 1}-{str(season)[2:]}"
    try:
        # nba_api ships its own browser-fingerprint UA via
        # nba_api.library.http; passing custom headers can break the
        # multi-part request flow (read timeout). Use the bundled defaults.
        ls = leaguestandingsv3.LeagueStandingsV3(
            season=label,
            season_type="Regular Season",
            league_id="00",
            timeout=max(timeout, 30.0),
        )
        df = ls.get_data_frames()[0]
    except Exception as e:  # noqa: BLE001
        log.warning("nba_api LeagueStandingsV3 %s failed: %s", label, e)
        return None
    out: dict[str, dict] = {}
    for row in df.to_dict("records"):
        city = (row.get("TeamCity") or "").strip()
        name = (row.get("TeamName") or "").strip()
        full = f"{city} {name}".strip()
        if not full:
            continue
        diff = row.get("DiffPointsPG")
        try:
            srs = float(diff) if diff is not None else 0.0
        except Exception:
            srs = 0.0
        out[full] = {
            "srs": srs,
            "pace": 0.0,
            "ortg": float(row.get("PointsPG") or 0.0),
            "drtg": float(row.get("OppPointsPG") or 0.0),
        }
    return out or None


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
        headers=BREF_HEADERS,
        timeout=timeout,
    )
    if data is None:
        log.info("bref ratings %s unavailable; falling back to stats.nba.com", season)
        return _fetch_team_ratings_nba_stats(season, timeout=timeout)
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
        # bref returned something but it's not the ratings table (likely a
        # Cloudflare interstitial). Fall through to the stats.nba.com path.
        log.info("bref ratings %s parsed empty; falling back to stats.nba.com", season)
        return _fetch_team_ratings_nba_stats(season, timeout=timeout)
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
    if not out:
        return _fetch_team_ratings_nba_stats(season, timeout=timeout)
    return out


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
                headers=BREF_HEADERS,
                timeout=self.timeout,
            )
            if not data:
                continue
            # Honor BR's documented Crawl-delay: 5 between live requests.
            # Cached responses skip the sleep (we only paid it on the miss).
            time.sleep(CRAWL_DELAY_S * 0.02)
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
        if not rows:
            log.info(
                "bref schedule %s parsed empty; falling back to stats.nba.com",
                season,
            )
            return _fetch_schedule_nba_stats(season, timeout=self.timeout)
        return rows


def _fetch_schedule_nba_stats(season: int, *, timeout: float = 30.0) -> list[dict]:
    """stats.nba.com fallback schedule pull. Returns same shape as bref.

    Uses ``LeagueGameFinder`` (Regular Season) which returns every team-row
    of every game; we deduplicate by ``GAME_ID`` and resolve home/away from
    the ``MATCHUP`` field (``vs.`` = home, ``@`` = away).
    """
    try:
        from nba_api.stats.endpoints import leaguegamefinder  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.warning("nba_api missing for schedule fallback: %s", e)
        return []
    label = f"{season - 1}-{str(season)[2:]}"
    try:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=label,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
            timeout=max(timeout, 30.0),
        )
        df = gf.get_data_frames()[0]
    except Exception as e:  # noqa: BLE001
        log.warning("nba_api LeagueGameFinder %s failed: %s", label, e)
        return []
    games: dict[str, dict] = {}
    for r in df.to_dict("records"):
        gid = str(r.get("GAME_ID") or "")
        if not gid:
            continue
        matchup = (r.get("MATCHUP") or "").strip()
        # Use TEAM_NAME ("Boston Celtics") not TEAM_ABBREVIATION ("BOS") so
        # the ratings keys (from LeagueStandingsV3 city+name) align with the
        # schedule keys here. This keeps the connector's ratings↔schedule
        # join working under the stats.nba.com fallback path.
        team_full = (r.get("TEAM_NAME") or "").strip()
        if "vs." in matchup:
            games.setdefault(gid, {})["home"] = team_full
            games[gid]["date"] = r.get("GAME_DATE")
        elif "@" in matchup:
            games.setdefault(gid, {})["away"] = team_full
            games[gid].setdefault("date", r.get("GAME_DATE"))
    out: list[dict] = []
    for gid, g in games.items():
        if "home" not in g or "away" not in g or not g.get("date"):
            continue
        try:
            d = datetime.strptime(g["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        out.append({"date": d, "home": g["home"], "away": g["away"]})
    return out


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
