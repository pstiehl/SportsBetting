"""FanDuel content-managed-page JSON connector.

FanDuel exposes the same public JSON its front-end uses, keyed off the
public ``_ak`` access token. No auth, no quota meaningful for our daily
use.

Endpoint: ``/api/content-managed-page?page=CUSTOM&customPageId={slug}``
returns ``attachments.events`` (dict) and ``attachments.markets`` (dict).
Moneyline markets have ``marketType == "MONEY_LINE"`` and
``marketName == "Moneyline"``.

Each runner carries ``winRunnerOdds.americanDisplayOdds.americanOdds``
(int) and ``result.type`` ("HOME" or "AWAY") which we use to assign sides.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

import httpx

from ..config import CACHE_DIR
from ..types import BookLine, Event, Sport, SourceProb
from .base import SourceConnector
from .bovada import devig_two_way

log = logging.getLogger(__name__)

BASE = "https://sbapi.mi.sportsbook.fanduel.com"
PAGE_URL = (
    BASE
    + "/api/content-managed-page?page=CUSTOM&customPageId={slug}"
    + "&pbHorizontal=false&_ak=FhMFpcPWXMeyZxOx&timezone=America%2FNew_York"
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Sport → FanDuel customPageId slug. Tennis: all common slugs 404 or return
# empty — we skip tennis on FanDuel and let Bovada cover it.
SPORT_SLUGS: dict[Sport, str] = {
    "mlb": "mlb",
    "nfl": "nfl",
    "nba": "nba",
    "nhl": "nhl",
}


class FanDuel(SourceConnector):
    name = "fanduel"
    version = "page-v1"
    is_live = True

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": UA, "Accept": "application/json"},
            follow_redirects=True,
        )

    def _get_json(self, client: httpx.Client, url: str) -> object | None:
        for attempt in (1, 2):
            try:
                r = client.get(url)
                if r.status_code != 200:
                    log.debug("fanduel GET %s → %s", url, r.status_code)
                    return None
                return r.json()
            except httpx.HTTPError as e:
                log.warning("fanduel GET %s attempt %d failed: %s", url, attempt, e)
        return None

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        out: list[Event] = []
        with self._client() as client:
            for tag, slug in SPORT_SLUGS.items():
                if sport is not None and tag != sport:
                    continue
                data = self._get_json(client, PAGE_URL.format(slug=slug))
                if not data:
                    continue
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(CACHE_DIR / f"fanduel_{tag}.json", "w") as f:
                        json.dump(data, f)
                except Exception:  # noqa: BLE001
                    pass
                out.extend(self._parse_page(data, tag))
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)
        return [e for e in out if start_dt <= e.commence_time <= end_dt]

    @staticmethod
    def parse_page(data: object, sport: Sport, now: datetime | None = None) -> list[Event]:
        """Public parse wrapper, used by tests."""
        return FanDuel._parse_page(data, sport, now)

    @staticmethod
    def _parse_page(data: object, sport: Sport, now: datetime | None = None) -> list[Event]:
        if not isinstance(data, dict):
            return []
        now = now or datetime.now(timezone.utc)
        attachments = data.get("attachments") or {}
        events = attachments.get("events") or {}
        markets = attachments.get("markets") or {}
        if not isinstance(events, dict) or not isinstance(markets, dict):
            return []

        # Index moneyline markets by eventId.
        ml_by_event: dict[int, dict] = {}
        for m in markets.values():
            if not isinstance(m, dict):
                continue
            mtype = (m.get("marketType") or "").upper()
            mname = (m.get("marketName") or "").lower()
            if mtype != "MONEY_LINE" and mname != "moneyline":
                continue
            ev_id = m.get("eventId")
            if ev_id is None:
                continue
            # Keep the first moneyline market per event (FanDuel sometimes
            # exposes multiple variants, e.g. with/without overtime; the
            # main one is sorted first by sortPriority).
            existing = ml_by_event.get(ev_id)
            if existing is None or (
                m.get("sortPriority", 10**9) < existing.get("sortPriority", 10**9)
            ):
                ml_by_event[ev_id] = m

        out: list[Event] = []
        for ev_id, market in ml_by_event.items():
            ev = events.get(ev_id) or events.get(str(ev_id))
            if ev is None:
                # Some pages key events by int; some by str. Try both.
                ev = next(
                    (e for e in events.values() if e.get("eventId") == ev_id),
                    None,
                )
            if ev is None:
                # Pull what we can from the market itself.
                ev = {"eventId": ev_id, "name": market.get("eventName") or ""}

            parsed = _parse_event(ev, market, sport, now)
            if parsed is not None:
                out.append(parsed)
        return out


def _runner_american(runner: dict) -> int | None:
    wro = runner.get("winRunnerOdds") or {}
    # New shape: nested under americanDisplayOdds
    ado = wro.get("americanDisplayOdds")
    if isinstance(ado, dict):
        if "americanOddsInt" in ado:
            try:
                return int(ado["americanOddsInt"])
            except Exception:  # noqa: BLE001
                pass
        if "americanOdds" in ado:
            v = ado["americanOdds"]
            try:
                return int(v)
            except Exception:  # noqa: BLE001
                pass
    # Older / spec-described shape: flat under winRunnerOdds
    if "americanOdds" in wro:
        v = wro["americanOdds"]
        try:
            return int(v)
        except Exception:  # noqa: BLE001
            pass
    return None


def _parse_event(ev: dict, market: dict, sport: Sport, now: datetime) -> Event | None:
    runners = market.get("runners") or []
    if len(runners) < 2:
        return None

    home_runner: dict | None = None
    away_runner: dict | None = None
    for r in runners:
        side = ((r.get("result") or {}).get("type") or "").upper()
        if side == "HOME":
            home_runner = r
        elif side == "AWAY":
            away_runner = r
    if home_runner is None or away_runner is None:
        # Fall back to runner order if side metadata missing
        away_runner, home_runner = runners[0], runners[1]

    home_american = _runner_american(home_runner)
    away_american = _runner_american(away_runner)
    if home_american is None or away_american is None:
        return None

    home_name = home_runner.get("runnerName") or ""
    away_name = away_runner.get("runnerName") or ""

    # Commence time — prefer market.marketTime, then event.openDate.
    commence = now
    for key in ("marketTime",):
        v = market.get(key)
        if isinstance(v, str):
            try:
                commence = datetime.fromisoformat(v.replace("Z", "+00:00"))
                break
            except Exception:  # noqa: BLE001
                pass
    if commence == now:
        od = ev.get("openDate")
        if isinstance(od, str):
            try:
                commence = datetime.fromisoformat(od.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                pass

    home_prob = devig_two_way(home_american, away_american)
    event_id = f"fanduel:{ev.get('eventId') or market.get('eventId')}"

    lines = [
        BookLine(book="fanduel", side="home", american=home_american, captured_at=now),
        BookLine(book="fanduel", side="away", american=away_american, captured_at=now),
    ]
    source_probs = [
        SourceProb(
            source="fanduel",
            home_win_prob=home_prob,
            captured_at=now,
            notes="fanduel moneyline devig",
        )
    ]
    return Event(
        event_id=event_id,
        sport=sport,
        league=sport.upper(),
        home=home_name,
        away=away_name,
        commence_time=commence,
        source_probs=source_probs,
        lines=lines,
    )
