"""Bovada public coupon JSON connector.

Bovada exposes the same public JSON endpoints its own front-end uses, no
auth required.

- Nav discovery: ``/services/sports/event/v2/nav/A/description/{sport}``
  returns ``{current, parents, children: [...]}`` where each child has a
  ``link`` slug, a human description, and a ``numEvents`` count.
- League coupon: ``/services/sports/event/coupon/events/A/description{link}``
  returns a list of groups, each with an ``events[]`` and a ``path[]``
  breadcrumb that we use to bucket tennis into atp vs wta.

For each event we extract the two-sided moneyline ("Head To Head" market),
devig it, and emit a single ``SourceProb`` for the home side plus
``BookLine`` entries for each side so the existing market-close consensus
blender can also use them.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Iterable

import httpx

from ..config import CACHE_DIR
from ..types import BookLine, Event, Sport, SourceProb
from .base import SourceConnector

log = logging.getLogger(__name__)

BASE = "https://www.bovada.lv"
NAV_URL = BASE + "/services/sports/event/v2/nav/A/description/{sport}"
COUPON_URL = (
    BASE
    + "/services/sports/event/coupon/events/A/description{link}"
    + "?marketFilterId=def&preMatchOnly=true&eventsLimit=200&lang=en"
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Top-level Bovada sport slugs → our Sport tag (for non-tennis sports the
# coupon endpoint is at a fixed top-league slug; tennis is discovered).
TEAM_SPORT_SLUGS: dict[str, Sport] = {
    "/baseball/mlb": "mlb",
    "/basketball/nba": "nba",
    "/football/nfl": "nfl",
    "/hockey/nhl": "nhl",
}

# Bovada uses these top-level sport names in the nav discovery endpoint.
TENNIS_NAV_SPORT = "tennis"


def _american_str_to_int(s: str | int | None) -> int | None:
    """Bovada uses 'EVEN', '+115', '-139'. Return int or None on garbage."""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = s.strip().upper()
    if s in ("EVEN", "EV"):
        return 100
    try:
        return int(s)
    except ValueError:
        return None


def _implied(american: int) -> float:
    if american == 0:
        return 0.5
    if american > 0:
        return 100.0 / (american + 100.0)
    return (-american) / ((-american) + 100.0)


def devig_two_way(home_american: int, away_american: int) -> float:
    """Return devigged home_win_prob from two-sided american odds."""
    ph = _implied(home_american)
    pa = _implied(away_american)
    s = ph + pa
    if s <= 0:
        return 0.5
    return max(0.001, min(0.999, ph / s))


def _tennis_sport_from_path(path: list[dict]) -> Sport | None:
    """Bucket Bovada tennis leaf-league into atp/wta. Skip doubles/mixed."""
    if not path:
        return None
    leaf_link = (path[0].get("link") or "").lower()
    # Only singles map to our Sport literal.
    if "doubles" in leaf_link or "mixed" in leaf_link:
        return None
    # Check women first — "men-s-singles" is a substring of "women-s-singles".
    if (
        "women-s-singles" in leaf_link
        or "women's-singles" in leaf_link
        or "/wta" in leaf_link
        or "itf-women-s" in leaf_link
    ):
        return "wta"
    if (
        "men-s-singles" in leaf_link
        or "men's-singles" in leaf_link
        or "/atp" in leaf_link
        or "itf-men-s" in leaf_link
    ):
        return "atp"
    return None


def _team_sport_from_code(sport_code: str) -> Sport | None:
    return {
        "BASE": "mlb",
        "BASK": "nba",
        "FOOT": "nfl",
        "HOCK": "nhl",
    }.get((sport_code or "").upper())


class Bovada(SourceConnector):
    name = "bovada"
    version = "coupon-v1"
    is_live = True

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout

    # ----- HTTP helpers -------------------------------------------------

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
                    log.warning("bovada GET %s → %s", url, r.status_code)
                    return None
                return r.json()
            except httpx.HTTPError as e:
                log.warning("bovada GET %s attempt %d failed: %s", url, attempt, e)
        return None

    # ----- discovery ---------------------------------------------------

    def _tennis_leaf_links(self, client: httpx.Client) -> list[str]:
        """Discover active tennis tournament leaf links via the nav endpoint."""
        data = self._get_json(client, NAV_URL.format(sport=TENNIS_NAV_SPORT))
        if not isinstance(data, dict):
            return []
        links: list[str] = []
        for child in data.get("children", []):
            num = child.get("numEvents") or 0
            link = child.get("link")
            if num > 0 and link:
                links.append(link)
        return links

    # ----- fetch -------------------------------------------------------

    def fetch_events(
        self,
        start: date,
        end: date,
        sport: Sport | None = None,
    ) -> list[Event]:
        out: list[Event] = []
        with self._client() as client:
            # Team sports
            for slug, tag in TEAM_SPORT_SLUGS.items():
                if sport is not None and tag != sport:
                    continue
                data = self._get_json(client, COUPON_URL.format(link=slug))
                if data:
                    self._cache(f"bovada_team_{tag}.json", data)
                    out.extend(self._parse_coupon(data, fallback_sport=tag))
            # Tennis (discover active tournaments)
            if sport is None or sport in ("atp", "wta"):
                for link in self._tennis_leaf_links(client):
                    data = self._get_json(client, COUPON_URL.format(link=link))
                    if data:
                        slug_safe = link.strip("/").replace("/", "_")
                        self._cache(f"bovada_{slug_safe}.json", data)
                        events = self._parse_coupon(data, fallback_sport=None)
                        if sport is not None:
                            events = [e for e in events if e.sport == sport]
                        out.extend(events)
        # Date filter
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(end, datetime.max.time(), tzinfo=timezone.utc)
        return [e for e in out if start_dt <= e.commence_time <= end_dt]

    def _cache(self, name: str, data: object) -> None:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_DIR / name, "w") as f:
                json.dump(data, f)
        except Exception as e:  # noqa: BLE001
            log.debug("bovada cache write failed for %s: %s", name, e)

    # ----- parser ------------------------------------------------------

    @staticmethod
    def parse_coupon(
        data: object,
        fallback_sport: Sport | None = None,
        now: datetime | None = None,
    ) -> list[Event]:
        """Public parse wrapper, used by tests."""
        return Bovada._parse_coupon(data, fallback_sport=fallback_sport, now=now)

    @staticmethod
    def _parse_coupon(
        data: object,
        fallback_sport: Sport | None = None,
        now: datetime | None = None,
    ) -> list[Event]:
        if not isinstance(data, list):
            return []
        now = now or datetime.now(timezone.utc)
        out: list[Event] = []
        for grp in data:
            if not isinstance(grp, dict):
                continue
            path = grp.get("path") or []
            tennis_sport = _tennis_sport_from_path(path)
            for ev in grp.get("events", []) or []:
                sport_code = ev.get("sport") or ""
                team_sport = _team_sport_from_code(sport_code)
                if sport_code == "TENN":
                    sport = tennis_sport
                else:
                    sport = team_sport or fallback_sport
                if sport is None:
                    continue
                parsed = _parse_event(ev, sport, now)
                if parsed is not None:
                    out.append(parsed)
        return out


def _parse_event(ev: dict, sport: Sport, now: datetime) -> Event | None:
    desc = ev.get("description") or ""
    competitors = ev.get("competitors") or []
    if len(competitors) < 2:
        return None
    # Determine home/away. Bovada uses `awayTeamFirst` to indicate display order;
    # the `home` slot is always the second listed when awayTeamFirst=True.
    away_first = bool(ev.get("awayTeamFirst"))
    names = [c.get("name") or "" for c in competitors[:2]]
    if away_first:
        away_name, home_name = names[0], names[1]
    else:
        home_name, away_name = names[0], names[1]

    # Find the moneyline market.
    home_american: int | None = None
    away_american: int | None = None
    for dg in ev.get("displayGroups") or []:
        for m in dg.get("markets") or []:
            mdesc = (m.get("description") or "").strip().lower()
            mkey = (m.get("descriptionKey") or "").strip().lower()
            if mdesc != "moneyline" and mkey != "head to head":
                continue
            outcomes = m.get("outcomes") or []
            if len(outcomes) < 2:
                continue
            # Map outcome.description → side by matching against home/away names.
            for o in outcomes:
                price = (o.get("price") or {}).get("american")
                american = _american_str_to_int(price)
                if american is None:
                    continue
                oname = (o.get("description") or "").strip()
                if oname == home_name:
                    home_american = american
                elif oname == away_name:
                    away_american = american
            if home_american is not None and away_american is not None:
                break
        if home_american is not None and away_american is not None:
            break

    if home_american is None or away_american is None:
        return None

    # Parse commence time (epoch ms).
    try:
        commence = datetime.fromtimestamp(int(ev["startTime"]) / 1000.0, tz=timezone.utc)
    except Exception:  # noqa: BLE001
        commence = now

    home_prob = devig_two_way(home_american, away_american)
    event_id = f"bovada:{ev.get('id') or ev.get('competitionId') or desc}"

    lines = [
        BookLine(book="bovada", side="home", american=home_american, captured_at=now),
        BookLine(book="bovada", side="away", american=away_american, captured_at=now),
    ]
    source_probs = [
        SourceProb(
            source="bovada",
            home_win_prob=home_prob,
            captured_at=now,
            notes="bovada moneyline devig",
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
