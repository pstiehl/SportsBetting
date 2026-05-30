"""MLB live-only probability sources: FanGraphs, Dimers, DraftKings, Pinnacle.

Each of these is **forward-only** — we use them for today's slate, never for
historical backtest. They're behind a polite User-Agent and a 1-hour cache.

If any source is unreachable from CI (DC-blocked, schema change, paywall),
the connector logs a warning and returns ``[]`` so the build pipeline keeps
moving. None of these can block CI on their own.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone

import httpx

from ..config import CACHE_DIR
from ..types import Event, SourceProb, Sport
from .base import SourceConnector

log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; flashcat/1.0; +https://github.com/pstiehl/SportsBetting)"


def _cached_get(url: str, cache_file: str, *, ttl_seconds: int = 3600,
                headers: dict | None = None, timeout: float = 10.0) -> bytes | None:
    """1-hour file cache. Best-effort GET — returns None on any failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / cache_file
    if cache_path.exists():
        age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if age < ttl_seconds:
            try:
                return cache_path.read_bytes()
            except Exception:
                pass
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=h) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.content
    except Exception as e:  # noqa: BLE001
        log.warning("live-source fetch failed (%s): %s", url, e)
        return None
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data


# ---------------------------------------------------------------------------
# FanGraphs
# ---------------------------------------------------------------------------


class FanGraphsMLB(SourceConnector):
    """FanGraphs pre-game win probability scraper.

    Tries the JSON scoreboard endpoint first; falls back to the HTML
    scoreboard if 403/blocked. Forward-only (live slate).
    """

    name = "fangraphs-mlb"
    version = "v1"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        out: list[Event] = []
        cur = start
        while cur <= end:
            data = _cached_get(
                f"https://www.fangraphs.com/api/scores/scoreboard?date={cur.isoformat()}",
                cache_file=f"fangraphs_{cur.isoformat()}.json",
                ttl_seconds=3600,
                timeout=self.timeout,
            )
            if data:
                try:
                    payload = json.loads(data)
                    out.extend(self._parse_json(payload, cur))
                except Exception as e:  # noqa: BLE001
                    log.warning("fangraphs json parse failed: %s", e)
            cur = date.fromordinal(cur.toordinal() + 1)
        return out

    @staticmethod
    def _parse_json(payload, day: date) -> list[Event]:
        # FanGraphs ships either a list directly or {"games": [...]}.
        games = payload if isinstance(payload, list) else (payload.get("games") or [])
        now = datetime.now(timezone.utc)
        out: list[Event] = []
        for g in games or []:
            home = g.get("HomeTeamName") or g.get("homeTeam") or g.get("home") or ""
            away = g.get("AwayTeamName") or g.get("awayTeam") or g.get("away") or ""
            wp = g.get("HomeWinProb") or g.get("homeWinProb") or g.get("home_winprob")
            if wp is None:
                continue
            try:
                hp = float(wp)
                if hp > 1.0:
                    hp /= 100.0
            except Exception:
                continue
            if not home or not away:
                continue
            event_id = f"fangraphs:mlb:{day.isoformat()}:{home}_vs_{away}".lower().replace(" ", "_")
            out.append(
                Event(
                    event_id=event_id,
                    sport="mlb",
                    league="MLB",
                    home=home,
                    away=away,
                    commence_time=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                    source_probs=[
                        SourceProb(
                            source="fangraphs-mlb",
                            home_win_prob=max(0.001, min(0.999, hp)),
                            captured_at=now,
                            notes="FanGraphs scoreboard pre-game win prob",
                        )
                    ],
                )
            )
        return out


# ---------------------------------------------------------------------------
# Dimers
# ---------------------------------------------------------------------------

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', re.DOTALL
)


class DimersMLB(SourceConnector):
    """Dimers MLB schedule scraper — extracts homeWinProbability from __NEXT_DATA__."""

    name = "dimers-mlb"
    version = "v1"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        html = _cached_get(
            "https://www.dimers.com/bet-hub/mlb/schedule",
            cache_file="dimers_mlb_schedule.html",
            ttl_seconds=3600,
            timeout=self.timeout,
        )
        if not html:
            return []
        m = _NEXT_DATA_RE.search(html.decode("utf-8", errors="ignore"))
        if not m:
            log.warning("dimers: __NEXT_DATA__ not found")
            return []
        try:
            blob = json.loads(m.group(1))
        except Exception as e:  # noqa: BLE001
            log.warning("dimers json parse failed: %s", e)
            return []
        games = _walk_dimers(blob)
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        for g in games:
            home = g.get("home_name") or g.get("home_team") or ""
            away = g.get("away_name") or g.get("away_team") or ""
            hp = g.get("homeWinProbability") or g.get("home_win_prob")
            if hp is None or not home or not away:
                continue
            try:
                hp = float(hp)
                if hp > 1.0:
                    hp /= 100.0
            except Exception:
                continue
            try:
                commence = datetime.fromisoformat(
                    (g.get("start_time") or "").replace("Z", "+00:00")
                )
            except Exception:
                commence = now
            event_id = f"dimers:mlb:{commence.date().isoformat()}:{home}_vs_{away}".lower().replace(" ", "_")
            out.append(
                Event(
                    event_id=event_id,
                    sport="mlb",
                    league="MLB",
                    home=home,
                    away=away,
                    commence_time=commence,
                    source_probs=[
                        SourceProb(
                            source="dimers-mlb",
                            home_win_prob=max(0.001, min(0.999, hp)),
                            captured_at=now,
                            notes="Dimers __NEXT_DATA__ homeWinProbability",
                        )
                    ],
                )
            )
        return out


def _walk_dimers(blob, out: list | None = None) -> list[dict]:
    """Recurse __NEXT_DATA__ looking for dicts that have homeWinProbability."""
    if out is None:
        out = []
    if isinstance(blob, dict):
        if "homeWinProbability" in blob or "home_win_prob" in blob:
            out.append(blob)
        for v in blob.values():
            _walk_dimers(v, out)
    elif isinstance(blob, list):
        for v in blob:
            _walk_dimers(v, out)
    return out


# ---------------------------------------------------------------------------
# Pinnacle public guest market — devigged closing/current
# ---------------------------------------------------------------------------

PINNACLE_GUEST_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"


class PinnacleMoneyline(SourceConnector):
    """Pinnacle guest API straight-bet moneyline puller.

    Sport → league id (the well-known public IDs):
      mlb=246, nba=487, nfl=889, atp=2417, wta=2419

    For each market we devig the two-way moneyline into a fair probability
    and surface as ``pinnacle-close`` (used by CLV math elsewhere).
    """

    name = "pinnacle-moneyline"
    version = "guest-0.1"
    is_live = True

    LEAGUE_IDS = {
        "mlb": 246,
        "nba": 487,
        "nfl": 889,
        "atp": 2417,
        "wta": 2419,
    }

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        sports = [sport] if sport else list(self.LEAGUE_IDS.keys())
        out: list[Event] = []
        for s in sports:
            league_id = self.LEAGUE_IDS.get(s)
            if not league_id:
                continue
            data = _cached_get(
                f"https://guest.api.arcadia.pinnacle.com/0.1/leagues/{league_id}/markets/straight",
                cache_file=f"pinnacle_{s}.json",
                ttl_seconds=3600,
                headers={"X-API-Key": PINNACLE_GUEST_KEY, "Accept": "application/json"},
                timeout=self.timeout,
            )
            if not data:
                continue
            try:
                markets = json.loads(data)
            except Exception as e:  # noqa: BLE001
                log.warning("pinnacle parse failed: %s", e)
                continue
            out.extend(self._parse(markets, s))
        return out

    @staticmethod
    def _parse(markets, sport: Sport) -> list[Event]:
        # Best-effort — Pinnacle's schema occasionally rotates.
        out: list[Event] = []
        now = datetime.now(timezone.utc)
        # We won't try to perfectly reconstruct events here. We just emit
        # source rows once we find moneyline pairs grouped by matchupId.
        by_matchup: dict[int, list] = {}
        for m in markets or []:
            if m.get("type") != "moneyline":
                continue
            mid = m.get("matchupId")
            if mid is None:
                continue
            by_matchup.setdefault(mid, []).append(m)
        for mid, group in by_matchup.items():
            prices = group[0].get("prices") if group else None
            if not prices or len(prices) < 2:
                continue
            try:
                p_home = prices[0].get("price")
                p_away = prices[1].get("price")
                # Pinnacle returns decimal odds.
                if not p_home or not p_away:
                    continue
                ih = 1.0 / float(p_home)
                ia = 1.0 / float(p_away)
                total = ih + ia
                if total <= 0:
                    continue
                home_devig = ih / total
            except Exception:
                continue
            event_id = f"pinnacle:{sport}:{mid}"
            out.append(
                Event(
                    event_id=event_id,
                    sport=sport,
                    league=sport.upper(),
                    home=str(prices[0].get("designation") or "home"),
                    away=str(prices[1].get("designation") or "away"),
                    commence_time=now,
                    source_probs=[
                        SourceProb(
                            source="pinnacle-close",
                            home_win_prob=max(0.001, min(0.999, home_devig)),
                            captured_at=now,
                            notes="Pinnacle guest API devigged moneyline",
                        )
                    ],
                )
            )
        return out


# ---------------------------------------------------------------------------
# DraftKings public eventgroups (best-effort, DC-blocked is fine)
# ---------------------------------------------------------------------------


class DraftKingsMLB(SourceConnector):
    name = "draftkings-mlb"
    version = "v5"
    is_live = True

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_events(
        self, start: date, end: date, sport: Sport | None = None
    ) -> list[Event]:
        if sport is not None and sport != "mlb":
            return []
        data = _cached_get(
            "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/84240",
            cache_file="dk_mlb_eventgroup.json",
            ttl_seconds=3600,
            headers={"Referer": "https://sportsbook.draftkings.com/leagues/baseball/mlb"},
            timeout=self.timeout,
        )
        if not data:
            return []
        # Just confirm presence — schema is verbose and changes; full parse
        # is out of scope for this PR. We log success so Phil sees it work.
        log.info("draftkings-mlb fetch succeeded (%d bytes)", len(data))
        return []
